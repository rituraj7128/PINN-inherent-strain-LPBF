"""Field-level verification of the PINN against the FE reference (reference mesh, unit strain):
relative L2 errors of the displacement fields (as-built, after cut C, after cut D, after removal with the best-fit
rigid-body motion removed) over all FE nodes, of the as-built stress over the solid elements, and the release
energies (work done by the release) of PINN and FE.  ->  results/field_errors.json"""
import json, glob, os, numpy as np, torch
import ispinn as isp, fe_reference as fe
g = isp.paper_cantilever(); m = isp.Material(); q = lambda *a: None
lh = g.H / round(g.H / 0.55); X8 = np.array(json.load(open("data/experiment_dinesh2025.json"))["probe_x_mm"])
NPZ = "results/fe_fields_ref.npz"
if not os.path.exists(NPZ):
    r = fe.run_fe(fe.boxes_from_geometry(g), layer_h=lh, E=m.E, nu=m.nu, h=(1.0, 1.0, lh / 2), cuts={"C": 60.0, "D": 54.0},
                  probe_x=X8, support=g.support.voigt(m.E, m.nu), log=q, return_fields=True)
    np.savez(NPZ, p=r["p"], u_tot=r["u_tot"], u_rm=r["u_rm"], u_cut_C=r["u_cut_C"], u_cut_D=r["u_cut_D"], S_el=r["S_el"],
             c=r["centroids"], e_C=r["energy_cut_C"], e_D=r["energy_cut_D"], e_rm=r.get("energy_removed", np.nan), dofs=r["dofs"])
F = np.load(NPZ); P = F["p"].T                                 # nodes (n, 3)
print("FE nodes", P.shape, "keys", list(F.keys()))


def nodal(a):                                                  # skfem nodal vector (3n) -> (n, 3)
    a = np.asarray(a)
    return a.reshape(-1, 3) if a.ndim == 1 and a.size == 3 * P.shape[0] else (a.T if a.shape[0] == 3 else a)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rigid_fit(d, X):                                            # least-squares rigid motion a + w x X fitted to d
    xc = X - X.mean(0); A = np.zeros((3 * len(X), 6)); A[0::3, 0] = A[1::3, 1] = A[2::3, 2] = 1
    A[0::3, 4], A[0::3, 5] = xc[:, 2], -xc[:, 1]; A[1::3, 3], A[1::3, 5] = -xc[:, 2], xc[:, 0]; A[2::3, 3], A[2::3, 4] = xc[:, 1], -xc[:, 0]
    c, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None); return (A @ c).reshape(-1, 3)


solid = ~g.is_support(torch.tensor(F["c"].T, dtype=torch.float32)).numpy()
out = {"fe_dofs": int(F["dofs"]), "seeds": {}}
for f in sorted(glob.glob("models/cantilever_seed*.pt")):
    s = isp.ISPINN.load(f, g, device="cpu", log=q); e0 = isp.EPS_UNIT; res = {}
    ut = s.displacement("onplate", P, e0); res["u_asbuilt"] = rel(ut, nodal(F["u_tot"]))
    for k in ("C", "D"):
        res[f"u_cut_{k}"] = rel(s.displacement(f"cut_{k}", P, e0, "release"), nodal(F[f"u_cut_{k}"]))
    ur = s.displacement("removed", P, e0, "release"); fr = nodal(F["u_rm"])                                      # FE release displacement (3-2-1 supported)
    d = ur - fr; res["u_removed_rigid_aligned"] = float(np.linalg.norm(d - rigid_fit(d, P)) / np.linalg.norm(fr - rigid_fit(fr, P)))
    Sp = s.stress("onplate", F["c"].T, e0); Sf = np.moveaxis(F["S_el"], -1, 0)
    res["stress_asbuilt_solid"] = rel(Sp[solid], Sf[solid])
    # release energies W = 1/2 int sigma_r : eps_r (N mm), stratified quadrature with 200k points
    torch.manual_seed(0); X, w = g.sample(200000, None, "cpu", True); st_ = g.is_support(X)
    for k in ("cut_C", "cut_D"):
        W = 0.0
        for i in range(0, X.shape[0], 10000):                  # chunks: the autograd graph of 200k points does not fit in memory
            Xg = X[i:i + 10000].clone().requires_grad_(True); u = s._u_rel(s.release_nets[k], s.specs[k], Xg)
            e = isp.sym(isp.grad_u(u, Xg)) / e0
            W += float((w[i:i + 10000] * 0.5 * (e * s._stress(e, st_[i:i + 10000])).sum((1, 2))).sum().detach()) * s.MPa * e0
        res[f"W_{k}_pinn"] = W; res[f"W_{k}_fe"] = float(-F["e_" + k[-1]]); res[f"W_{k}_ratio"] = W / res[f"W_{k}_fe"]
    out["seeds"][os.path.basename(f)] = res; print(f, {k: round(v, 4) for k, v in res.items()})
keys = list(next(iter(out["seeds"].values())))
out["mean"] = {k: float(np.mean([v[k] for v in out["seeds"].values()])) for k in keys}
json.dump(out, open("results/field_errors.json", "w"), indent=1); print("mean", out["mean"])
