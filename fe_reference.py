"""
Independent finite-element reference for the SAME layer-wise inherent-strain model that the PINN solves.
Used to verify the PINN (solver accuracy) separately from the model assumptions.

Linear elasticity, 8-node hexahedra (scikit-fem), quiet-element layer activation:
  * build: layer k is activated, Δu_k solves K_k Δu = ∫ Bᵀ C ε*_k  (ε* = -eps I in layer k only),
           stress of active material accumulates; inactive material has stiffness 1e-6.
  * release: K u_r = -∫ Bᵀ S0 with the new supports (partial cut or free).
Supports use the same homogenised comb law as the PINN (ispinn.SupportModel.voigt).
Options: S_func  - drive the release by an external as-built stress (used to split PINN build/release error)
         return_fields - also return nodal fields, element stresses and centroids.

Units mm, MPa. Results are per unit inherent strain EPS_UNIT = 1e-3 (scale linearly).
"""
import time, gc
import numpy as np
from skfem import MeshHex, Basis, ElementVector, ElementHex1, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import ddot, sym_grad, trace, eye
import pyamg

def _rigid_modes(basis):
    p, nd = basis.mesh.p, basis.nodal_dofs
    B = np.zeros((basis.N, 6))
    x, y, z = p
    for i in range(3):
        B[nd[i], i] = 1.0
    B[nd[1], 3], B[nd[2], 3] = -z, y
    B[nd[0], 4], B[nd[2], 4] = z, -x
    B[nd[0], 5], B[nd[1], 5] = -y, x
    return B

def _solve(K, f, D, B, tol=1e-9):
    A, b, x, I = condense(K, f, D=D)
    A = A.tocsr()
    ml = pyamg.smoothed_aggregation_solver(A, B=B[I], max_coarse=500)
    xi = ml.solve(b, tol=tol, accel="cg", maxiter=500)
    del ml, A; gc.collect()
    out = np.zeros(K.shape[0]); out[I] = xi
    return out

EPS_UNIT = 1e-3


def _voigt(e):
    return [e[0][0], e[1][1], e[2][2], 2 * e[1][2], 2 * e[0][2], 2 * e[0][1]]


def _apply(Cm, e):
    ev = _voigt(e)
    s = [sum(Cm[i, j] * ev[j] for j in range(6) if Cm[i, j] != 0) for i in range(6)]
    s = [si if not isinstance(si, int) else 0 * e[0][0] for si in s]
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


def run_fe(boxes, layer_h=1.1, E=200e3, nu=0.3, h=(1.0, 1.0, 0.5), cuts=None, probe_x=None,
           probe_y=0.0, support=None, log=print, return_fields=False, S_func=None):
    """boxes: list of (lo, hi, stiffness_factor) covering the part (rectangular bounding domain is meshed,
    elements whose centroid is in no box are removed). cuts: dict name -> x_cut (plate bond released for x < x_cut).
    Returns dict with top-line z displacement per scenario (mm per unit strain) and on-plate stress."""
    t0 = time.time()
    lo = np.min([b[0] for b in boxes], 0); hi = np.max([b[1] for b in boxes], 0)
    axes = [np.linspace(lo[i], hi[i], int(round((hi[i] - lo[i]) / h[i])) + 1) for i in range(3)]
    mesh = MeshHex.init_tensor(*axes)
    c = mesh.p[:, mesh.t].mean(1)                                    # element centroids (3, nel)
    mat = np.full(mesh.t.shape[1], -1)                              # 0 solid, 1 support
    for blo, bhi, kind in boxes:                                    # later boxes override
        m = np.all((c >= np.array(blo)[:, None]) & (c <= np.array(bhi)[:, None]), 0)
        mat[m] = 0 if kind == "solid" else 1
    if np.any(mat < 0):
        mesh = mesh.remove_elements(np.nonzero(mat < 0)[0]); c = mesh.p[:, mesh.t].mean(1); mat = mat[mat >= 0]
    Csup = support
    basis = Basis(mesh, ElementVector(ElementHex1()), intorder=2)
    nqp = basis.X.shape[1]
    B = _rigid_modes(basis)
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

    def C(e):
        return 2 * mu * e + lam * eye(trace(e), 3)

    Ce0_sup = _apply(Csup, -np.eye(3)[:, :, None, None] * np.ones((1, 1, 1, 1)))[:, :, 0, 0]   # C_sup : (-I)

    @BilinearForm
    def stiff(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return w.ks * ddot(C(eu), ev) + w.kp * ddot(_apply(Csup, eu), ev)

    @LinearForm
    def eig_load(v, w):          # ∫ C ε* : ε(v), ε* = -m * EPS_UNIT * I
        ev = sym_grad(v)
        sol = -(2 * mu + 3 * lam) * trace(ev)
        sup = sum(Ce0_sup[i, j] * ev[i][j] for i in range(3) for j in range(3))
        return w.m * EPS_UNIT * (w.ks * sol + w.kp * sup)

    @LinearForm
    def stress_load(v, w):       # -∫ S0 : ε(v)
        return -ddot(w.S, sym_grad(v))

    z0, H = lo[2], hi[2] - lo[2]
    nl = max(1, int(round(H / layer_h)))
    zk = z0 + H * np.arange(nl + 1) / nl
    zc = c[2]
    plate = basis.get_dofs(lambda x: x[2] < z0 + 1e-9).all()
    nodes = basis.nodal_dofs                                         # (3, nnodes)
    pz = mesh.p[2]
    u_tot = np.zeros(basis.N)
    S = np.zeros((3, 3, mesh.t.shape[1], nqp))
    rep = lambda a: np.repeat(np.asarray(a, float)[:, None], nqp, 1)
    ks_all, kp_all = rep(mat == 0), rep(mat == 1)
    if S_func is not None:          # external as-built stress (MPa) at quadrature points: skip the build
        qp = np.asarray(basis.global_coordinates().value)        # (3, nel, nqp)
        S = np.asarray(S_func(qp.reshape(3, -1).T)).transpose(1, 2, 0).reshape(3, 3, *qp.shape[1:])
        nl = 0
    for k in range(1, nl + 1):
        active = zc <= zk[k] + 1e-9
        layer = (zc > (zk[k - 1] if k > 1 else -1e9)) & active
        K = None; gc.collect()                                       # free the previous step before assembling (memory)
        K = asm(stiff, basis, ks=rep((mat == 0) & active), kp=rep((mat == 1) & active))
        f = asm(eig_load, basis, ks=ks_all, kp=kp_all, m=rep(layer))
        inact_nodes = np.setdiff1d(np.arange(mesh.p.shape[1]), np.unique(mesh.t[:, active]))
        du = _solve(K, f, np.unique(np.r_[plate, nodes[:, inact_nodes].ravel()]), B)
        g = basis.interpolate(du).grad                               # (3,3,nel,nqp)
        eps = 0.5 * (g + np.swapaxes(g, 0, 1))
        e0 = -np.repeat(layer[:, None].astype(float), nqp, 1) * EPS_UNIT
        el = eps.copy(); el[0, 0] -= e0; el[1, 1] -= e0; el[2, 2] -= e0
        tr = el[0, 0] + el[1, 1] + el[2, 2]
        dS = np.where((mat == 0)[None, None, :, None], 2 * mu * el + lam * tr * np.eye(3)[:, :, None, None],
                      _apply(Csup, el))
        S += np.where(active[None, None, :, None], dS, 0.0)
        nm = (pz <= zk[k] + 1e-9)                                   # nodes existing at step k
        u_tot[nodes[:, nm]] += du[nodes[:, nm]]
        log(f"  FE layer {k}/{nl}  ({time.time()-t0:.0f}s)")

    K = asm(stiff, basis, ks=ks_all, kp=kp_all)
    f = asm(stress_load, basis, S=S)
    out = {"mesh_nodes": mesh.p.shape[1], "dofs": basis.N, "layers": nl}
    # top line
    top = (np.abs(mesh.p[2] - hi[2]) < 1e-9) & (np.abs(mesh.p[1] - probe_y) < 1e-9)
    order = np.argsort(mesh.p[0, top]); tx = mesh.p[0, top][order]; tn = np.nonzero(top)[0][order]
    px = np.asarray(probe_x if probe_x is not None else tx)
    line = lambda u: np.interp(px, tx, u[nodes[2, tn]])
    out["asbuilt_dz"] = line(u_tot)
    # stress at top surface elements along the line (element nearest)
    topel = np.nonzero((c[2] > hi[2] - h[2]) & (np.abs(c[1] - probe_y) < h[1]))[0]
    sx = S[0, 0, topel].mean(1); ex = c[0, topel]; o = np.argsort(ex)
    out["asbuilt_sxx_top"] = np.interp(px, ex[o], sx[o])
    def _vm(T):
        tr = (T[0, 0] + T[1, 1] + T[2, 2]) / 3
        dev = T - tr * np.eye(3)[:, :, None]
        return np.sqrt(1.5 * (dev * dev).sum((0, 1)))
    topS = S[:, :, topel].mean(3)                                   # (3,3,ntop) element-mean stress
    out["asbuilt_vm_top"] = np.interp(px, ex[o], _vm(topS)[o])
    def _rel_stress(ur):
        g = basis.interpolate(ur).grad; e = 0.5 * (g + np.swapaxes(g, 0, 1))
        tr = e[0, 0] + e[1, 1] + e[2, 2]
        dS = 2 * mu * e + lam * tr * np.eye(3)[:, :, None, None]   # top elements are solid
        return (S + dS)[:, :, topel].mean(3)
    for name, xcut in (cuts or {}).items():
        if isinstance(xcut, (tuple, list)):                       # keep the plate bond only for lo <= x <= hi
            lo_k, hi_k = xcut
            D = basis.get_dofs(lambda x: (x[2] < z0 + 1e-9) & (x[0] >= lo_k - 1e-9) & (x[0] <= hi_k + 1e-9)).all()
        else:
            D = basis.get_dofs(lambda x: (x[2] < z0 + 1e-9) & (x[0] >= xcut - 1e-9)).all()
        ur = _solve(K, f, D, B)
        out[f"cut_{name}"] = line(ur)
        if return_fields: out[f"u_cut_{name}"] = ur[nodes].copy()
        out[f"energy_cut_{name}"] = float(-0.5 * f @ ur)
        Sc = _rel_stress(ur)
        out[f"vm_top_cut_{name}"] = np.interp(px, ex[o], _vm(Sc)[o])
        out[f"sxx_top_cut_{name}"] = np.interp(px, ex[o], Sc[0, 0][o])
    # free: 3-2-1 constraint, then the shape is compared after removing a best-fit line
    pn = mesh.p
    a = np.argmin(np.sum((pn - lo[:, None]) ** 2, 0))
    b = np.argmin(np.sum((pn - np.array([hi[0], lo[1], lo[2]])[:, None]) ** 2, 0))
    cc = np.argmin(np.sum((pn - np.array([lo[0], hi[1], lo[2]])[:, None]) ** 2, 0))
    D = np.r_[nodes[:, a], nodes[1:, b], nodes[2, cc]]
    u_rm = _solve(K, f, D, B)
    out["removed"] = line(u_rm)
    if return_fields:
        out["p"] = mesh.p; out["u_tot"] = u_tot[nodes].copy(); out["u_rm"] = u_rm[nodes].copy()
        out["S_el"] = S.mean(3); out["centroids"] = c
    out["time_s"] = time.time() - t0
    return out


def boxes_from_geometry(geom):
    """FE box list from an ispinn.Geometry (support boxes first so part boxes override)."""
    sup = [(b.lo, b.hi, "support") for b in geom.boxes if b.role == "support"]
    part = [(b.lo, b.hi, "solid") for b in geom.boxes if b.role == "part"]
    return sup + part
