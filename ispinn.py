"""
ispinn — energy-based physics-informed neural network (PINN) for the layer-wise inherent strain method
of laser powder bed fusion (LPBF): residual stress and distortion on the build plate, after a partial
cut and after removal from the plate.

Physics (Section 2 of the paper)
    Build step k (k = 1..N lumped layers): layer k receives the inherent strain -eps* I, and the
    displacement increment du_k minimises the elastic energy of the active body (Eq. 2).
    Release (cut or removal): the release displacement u_r minimises Eq. 4, driven by the as-built
    stress sigma_0. No simulation or measurement data enter any loss.

Solver features for thin, support-dominated parts (Section 2.4)
    1. layer-interface enrichment: the build network of step k also receives relu(z - z_{k-1}) / h
    2. release features: interface ramps relu(x_i - c)/(H/4) and cut-tip features (distance, sqrt(r/H))
    3. stratified quadrature (+ extra points around each cut tip)
    4. Adam warm start followed by full-batch L-BFGS in every release stage

Calibration (Section 2.5): the problem is linear in eps*, so the networks are trained once for the unit
strain EPS_UNIT = 1e-3 and every field is scaled by eps*/EPS_UNIT. eps* is fitted in closed form.

Units: mm, MPa. Inherent strain is given as a positive number (= shrinkage).
"""
import time
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import torch.nn as nn

EPS_UNIT = 1e-3   # unit inherent strain of the basis solution


# =====================================================================================
# Settings
# =====================================================================================
@dataclass
class Material:
    name: str = "IN718"
    E: float = 200e3            # MPa
    nu: float = 0.30


@dataclass
class SolverSettings:
    layer_h: float = 0.55       # lumped layer thickness (mm); eps* is tied to it
    width: int = 64             # hidden units per layer
    depth: int = 4              # hidden layers
    n_pts: int = 4000           # collocation points per build iteration
    iters_build: int = 4000     # Adam iterations per build layer
    lr: float = 3e-3            # Adam learning rate, build
    n_cloud: int = 60000        # fixed point cloud for the release stages
    batch: int = 4000           # Adam mini-batch, release
    iters_release: int = 3000   # Adam iterations per release
    lr_release: float = 1e-2    # Adam learning rate, release
    lbfgs_release: int = 1500   # full-batch L-BFGS iterations per release
    tip_n: int = 8000           # extra points around each cut tip
    tip_r: float = 6.0          # radius (mm) of the refined zone around a cut tip
    rb_weight: float = 10.0     # rigid-body penalty after full removal
    seed: int = 0
    # feature switches (all True in the paper; switched off one at a time only for the ablation study)
    enrich: bool = True         # layer-interface feature relu(z - z_{k-1})/h in the build networks
    rfeat: bool = True          # interface ramps and cut-tip features in the release networks
    stratified: bool = True     # stratified quadrature (False: uniform points in the bounding box)

    @staticmethod
    def preset(name="paper"):
        """'paper': settings of the paper (GPU, ~50 min per geometry on a 6 GB RTX A2000).
        'cpu': reduced settings for a CPU check (~1 h). 'smoke': seconds, only tests that the code runs."""
        s = SolverSettings()
        if name == "cpu":
            s.n_pts, s.batch, s.n_cloud, s.tip_n, s.lbfgs_release = 2000, 2000, 36000, 6000, 1200
        elif name == "smoke":
            s.n_pts, s.batch, s.n_cloud, s.tip_n = 600, 600, 3000, 500
            s.iters_build, s.iters_release, s.lbfgs_release = 20, 30, 50
        elif name != "paper":
            raise ValueError(name)
        return s


# =====================================================================================
# Geometry: union of axis-aligned boxes on the build plate (z = z0)
# =====================================================================================
@dataclass
class SupportModel:
    """Homogenised comb support (Section 2.2): thin walls normal to x, thickness t, pitch p, height h.
    Plane-stress stiffness in the wall plane (yz) scaled by phi = t/p; almost no stiffness for normal stress
    along x or for xy shear; xz shear stiffness from bending of clamped walls."""
    t: float = 1.0
    p: float = 2.0
    h: float = 8.4

    def voigt(self, E, nu):
        """6x6 stiffness, Voigt order (xx, yy, zz, yz, xz, xy)."""
        G, phi = E / (2 * (1 + nu)), self.t / self.p
        C = np.zeros((6, 6))
        Q = phi * E / (1 - nu ** 2)
        C[1, 1] = C[2, 2] = Q; C[1, 2] = C[2, 1] = nu * Q
        C[0, 0] = C[5, 5] = 1e-3 * phi * E
        C[3, 3] = phi * G
        C[4, 4] = min(E * self.t ** 3 / (self.h ** 2 * self.p), phi * G)
        return C


@dataclass
class Box:
    lo: tuple
    hi: tuple
    role: str = "part"          # "part" (solid) or "support" (homogenised comb)


class Geometry:
    def __init__(self, boxes, name="part", support=None):
        self.boxes, self.name = list(boxes), name
        self.support = support or SupportModel()
        self.lo = torch.tensor(np.min([b.lo for b in self.boxes], 0), dtype=torch.float32)
        self.hi = torch.tensor(np.max([b.hi for b in self.boxes], 0), dtype=torch.float32)
        self.z0, self.H = float(self.lo[2]), float(self.hi[2] - self.lo[2])
        self.Lmax = float((self.hi - self.lo)[:2].max())
        self._blo = torch.tensor([b.lo for b in self.boxes], dtype=torch.float32)
        self._bhi = torch.tensor([b.hi for b in self.boxes], dtype=torch.float32)
        self._part = torch.tensor([b.role == "part" for b in self.boxes])

    def describe(self):
        return {"name": self.name, "boxes": [asdict(b) for b in self.boxes], "support": asdict(self.support)}

    def _in_box(self, X):
        lo, hi = self._blo.to(X.device), self._bhi.to(X.device)
        return ((X[:, None, :] >= lo - 1e-9) & (X[:, None, :] <= hi + 1e-9)).all(-1)

    def _masks(self, X):
        """(inside solid part, inside homogenised support); solid boxes override support boxes."""
        m, isp = self._in_box(X), self._part.to(X.device)
        part = (m & isp).any(1)
        return part, (m & ~isp).any(1) & ~part

    def inside(self, X):
        p, s = self._masks(X)
        return p | s

    def is_support(self, X):
        return self._masks(X)[1]

    def interface_planes(self):
        """(axis, coordinate) of every box face strictly inside the bounding box (material interfaces)."""
        out = []
        for a in range(3):
            lo, hi = float(self.lo[a]), float(self.hi[a])
            cs = sorted({round(float(v), 6) for b in self.boxes for v in (b.lo[a], b.hi[a])})
            out += [(a, c) for c in cs if lo + 1e-6 < c < hi - 1e-6]
        return out

    def _owner(self, X):
        """index of the box a point belongs to: first solid box containing it, else first support box."""
        m, isp, n = self._in_box(X), self._part.to(X.device), len(self.boxes)
        ar = torch.arange(n, device=X.device)
        part_idx = torch.where(m & isp, ar, n).min(1).values
        sup_idx = torch.where(m & ~isp, ar, n).min(1).values
        return torch.where(part_idx < n, part_idx, sup_idx)

    def sample(self, n, zmax=None, device="cpu", stratified=True):
        """Stratified quadrature (Section 2.4): box i gets a share of the n points proportional to
        volume / smallest dimension; weights = volume / count per box, so the quadrature is unbiased.
        stratified=False: uniform points in the bounding box (standard PINN), equal weights."""
        if not stratified:
            lo, hi = self.lo.clone(), self.hi.clone()
            if zmax is not None:
                hi[2] = min(float(hi[2]), float(zmax))
            Xs, got, gen = [], 0, 0
            while got < n:
                C = lo + torch.rand(2 * n, 3) * (hi - lo); gen += C.shape[0]
                C = C[self.inside(C)]; Xs.append(C); got += C.shape[0]
            V = float(torch.prod(hi - lo)) * got / gen          # volume of the (active) part, Monte Carlo
            return torch.cat(Xs)[:n].to(device), torch.full((n,), V / n).to(device)
        boxes = []
        for i, b in enumerate(self.boxes):
            lo = torch.tensor(b.lo, dtype=torch.float32); hi = torch.tensor(b.hi, dtype=torch.float32)
            if zmax is not None:
                hi[2] = min(float(hi[2]), float(zmax))
            if float(hi[2]) > float(lo[2]) + 1e-9:
                boxes.append((i, lo, hi, float(torch.prod(hi - lo) / (hi - lo).min())))
        tot = sum(b[3] for b in boxes)
        Xs, ws = [], []
        for i, lo, hi, a in boxes:
            C = lo + torch.rand(max(16, int(n * a / tot)), 3) * (hi - lo)
            keep = self._owner(C) == i
            if not bool(keep.any()):
                continue
            V = float(torch.prod(hi - lo)) * float(keep.float().mean())
            Xs.append(C[keep]); ws.append(torch.full((int(keep.sum()),), V / int(keep.sum())))
        return torch.cat(Xs).to(device), torch.cat(ws).to(device)


def paper_cantilever(L=72.0, W=12.0, H=9.0, beam_t=2.6, fixed_block=10.0, end_wall=3.0, tooth_t=1.0, pitch=2.0):
    """Cantilever of Dinesh & Sahu, Mater. Today Commun. 42 (2025) 111512 (dimensions: Table 1 of the paper).
    x = 0 at the free end: 2.6 mm solid beam on 6.4 mm comb teeth, 10 mm solid block at the fixed end,
    3 mm solid wall at the free end. Their Fig. 2(b): a 9 mm cantilever on a 2 mm solid extrusion of the base;
    the extrusion is treated as part of the rigid build plate (z = 0 is its top, where the wire-EDM cut is made).
    With layer_h = 0.55 the 9 mm part is split into round(9/0.55) = 16 layers of 0.5625 mm."""
    h_t = H - beam_t
    boxes = [Box((0, -W / 2, h_t), (L, W / 2, H), "part"),                      # beam
             Box((L - fixed_block, -W / 2, 0), (L, W / 2, h_t), "part"),        # fixed-end block
             Box((0, -W / 2, 0), (L - fixed_block, W / 2, h_t), "support"),     # comb teeth (homogenised)
             Box((0, -W / 2, 0), (end_wall, W / 2, h_t), "part")]               # free-end wall
    return Geometry(boxes, "cantilever_DineshSahu2025", SupportModel(t=tooth_t, p=pitch, h=h_t))


def twin_cantilever(L=72.0, W=12.0, H=11.0, beam_t=2.6, pillar=(31.0, 41.0), end_wall=3.0, tooth_t=1.0, pitch=2.0):
    """Twin cantilever of Section 3.2: same section, central solid pillar, two 3 mm end walls."""
    h_t = H - beam_t
    boxes = [Box((0, -W / 2, h_t), (L, W / 2, H), "part"),
             Box((pillar[0], -W / 2, 0), (pillar[1], W / 2, h_t), "part"),
             Box((0, -W / 2, 0), (end_wall, W / 2, h_t), "part"),
             Box((L - end_wall, -W / 2, 0), (L, W / 2, h_t), "part"),
             Box((end_wall, -W / 2, 0), (pillar[0], W / 2, h_t), "support"),
             Box((pillar[1], -W / 2, 0), (L - end_wall, W / 2, h_t), "support")]
    return Geometry(boxes, "twin_cantilever_72", SupportModel(t=tooth_t, p=pitch, h=h_t))


# Scenarios used in the paper
SCEN_CANTILEVER = {"onplate": {"type": "onplate"},
                   "cut_C": {"type": "cut", "axis": 0, "from": "min", "pos": 60.0},   # released for x < 60
                   "cut_D": {"type": "cut", "axis": 0, "from": "min", "pos": 54.0},
                   "removed": {"type": "full"}}
SCEN_TWIN = {"onplate": {"type": "onplate"},
             "cut": {"type": "cut", "axis": 0, "keep": [29.0, 41.85]}}  # bond kept as in the Simufact run (pillar 31-41 mm)


# =====================================================================================
# Network and mechanics helpers
# =====================================================================================
class Net(nn.Module):
    """Fully connected network x -> u (3 outputs); inputs scaled to [-1, 1]; optional extra features."""
    def __init__(self, lo, hi, width, depth, extra=0):
        super().__init__()
        self.register_buffer("lo", lo.clone())
        self.register_buffer("hi", hi.clone())
        layers, d = [], 3 + extra
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X, feat=None):
        x = 2 * (X - self.lo) / (self.hi - self.lo) - 1
        return self.mlp(x if feat is None else torch.cat([x, feat], 1))


def grad_u(u, X):
    return torch.stack([torch.autograd.grad(u[:, i].sum(), X, create_graph=True)[0] for i in range(3)], 1)


def sym(G):
    return 0.5 * (G + G.transpose(1, 2))


def von_mises(S):
    s = S - np.trace(S, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * (s * s).sum((-2, -1)))


def top_line(geom, xs, y=0.0, z_off=1e-3):
    """Points on the top surface centre line (where distortion and stress are measured)."""
    return np.array([[float(x), y, float(geom.hi[2]) - z_off] for x in xs])


def detrend(y, x):
    """Remove the best-fit line (shape after removal, as in the CT comparison)."""
    A = np.vstack([x, np.ones_like(x)]).T
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def calibrate_eps(u_unit, u_meas):
    """Closed-form least-squares calibration (Eq. 9): eps* = EPS_UNIT (u0 . u_meas) / (u0 . u0)."""
    u_unit, u_meas = np.asarray(u_unit, float), np.asarray(u_meas, float)
    return float(u_unit @ u_meas / (u_unit @ u_unit)) * EPS_UNIT


# =====================================================================================
# Solver
# =====================================================================================
class ISPINN:
    """Layer-wise inherent-strain PINN for one geometry, trained for the unit strain EPS_UNIT.

    Scenarios (dict name -> spec):
        {"type": "onplate"}                                      as built, on the plate
        {"type": "cut", "axis": 0, "from": "min", "pos": 60}     bond released for x < 60
        {"type": "cut", "axis": 0, "keep": [31, 41]}             bond kept only for 31 <= x <= 41
        {"type": "full"}                                         removed from the plate
    """

    def __init__(self, geom, material=Material(), settings=SolverSettings(), device=None, log=print):
        self.g, self.m, self.s, self.log = geom, material, settings, log
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        nu = material.nu
        self.lam, self.mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))   # normalised by E
        self.MPa = material.E * EPS_UNIT                                          # stress unit (MPa)
        self.Csup = torch.tensor(geom.support.voigt(1.0, nu), dtype=torch.float32, device=self.dev)
        n_layers = max(1, int(round(geom.H / settings.layer_h)))
        self.zk = [geom.z0 + geom.H * k / n_layers for k in range(n_layers + 1)]  # layer boundaries
        self.U_build = EPS_UNIT * geom.H                                          # displacement scales
        self.U_rel = EPS_UNIT * geom.Lmax ** 2 / geom.H
        self.build_nets, self.release_nets, self.specs, self.history = [], {}, {}, {}
        self._cloud = None

    # ---------------- constitutive law ----------------
    def _stress(self, e, sup):
        """sigma = C : e (normalised by E): isotropic solid, homogenised comb where sup is True."""
        tr = e.diagonal(dim1=1, dim2=2).sum(1)
        s_iso = self.lam * tr[:, None, None] * torch.eye(3, device=e.device) + 2 * self.mu * e
        if not bool(sup.any()):
            return s_iso
        ev = torch.stack([e[:, 0, 0], e[:, 1, 1], e[:, 2, 2], 2 * e[:, 1, 2], 2 * e[:, 0, 2], 2 * e[:, 0, 1]], 1)
        sv = ev @ self.Csup.T
        s_sup = torch.stack([torch.stack([sv[:, 0], sv[:, 5], sv[:, 4]], 1),
                             torch.stack([sv[:, 5], sv[:, 1], sv[:, 3]], 1),
                             torch.stack([sv[:, 4], sv[:, 3], sv[:, 2]], 1)], 1)
        return torch.where(sup[:, None, None], s_sup, s_iso)

    def _net(self, extra=0):
        return Net(self.g.lo, self.g.hi, self.s.width, self.s.depth, extra).to(self.dev)

    @staticmethod
    def _freeze(net):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    # ---------------- build ----------------
    def _u_build(self, net, X, k):
        """Displacement increment of build step k (Eq. 7, left), with the layer-interface feature (Eq. 8)."""
        feat = torch.relu(X[:, 2:3] - self.zk[k - 1]) / (self.zk[1] - self.zk[0]) * float(self.s.enrich)
        return self.U_build * ((X[:, 2:3] - self.g.z0) / self.g.H) * net(X, feat)

    def _layer_mask(self, X, k):
        z = X[:, 2]
        lo = self.zk[k - 1] if k > 1 else -1e9
        return ((z > lo) & (z <= self.zk[k] + 1e-9)).float()

    def train_build(self):
        s, prev = self.s, None
        I = torch.eye(3, device=self.dev)
        for k in range(1, len(self.zk)):
            net = self._net(1)
            if prev is not None:
                net.load_state_dict(prev.state_dict())          # warm start from the previous layer
            opt = torch.optim.Adam(net.parameters(), lr=s.lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s.iters_build)
            hist, t0 = [], time.time()
            for _ in range(s.iters_build):
                X, w = self.g.sample(s.n_pts, self.zk[k], self.dev, s.stratified)
                X.requires_grad_(True)
                stiff = self.g.is_support(X)
                e = sym(grad_u(self._u_build(net, X, k), X)) / EPS_UNIT
                e0 = -self._layer_mask(X, k)[:, None, None] * I                  # inherent strain of layer k
                loss = (w * 0.5 * ((e - e0) * self._stress(e - e0, stiff)).sum((1, 2))).sum() / w.sum()   # Eq. 2
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                hist.append(loss.item())
            self.history[f"build_{k}"] = hist
            self.log(f"[{self.g.name}] build layer {k:2d}/{len(self.zk)-1}  energy {hist[-1]:.3e}  ({time.time()-t0:.0f}s)")
            self.build_nets.append(self._freeze(net))
            prev = net

    def asbuilt(self, X, bs=4000):
        """Unit-strain as-built displacement (mm) and stress (normalised by E*EPS_UNIT), Eq. 3."""
        us, Ss = [], []
        I = torch.eye(3, device=X.device)
        for i in range(0, X.shape[0], bs):
            Xb = X[i:i + bs].clone().requires_grad_(True)
            stiff = self.g.is_support(Xb)
            u = torch.zeros_like(Xb)
            S = torch.zeros(Xb.shape[0], 3, 3, device=Xb.device)
            for j, net in enumerate(self.build_nets, start=1):
                inside = (Xb[:, 2] <= self.zk[j] + 1e-9).float()
                uj = self._u_build(net, Xb, j)
                ej = sym(grad_u(uj, Xb)) / EPS_UNIT
                e0 = -self._layer_mask(Xb, j)[:, None, None] * I
                S = S + inside[:, None, None] * self._stress(ej - e0, stiff)
                u = u + inside[:, None] * uj
            us.append(u.detach()); Ss.append(S.detach())
        return torch.cat(us), torch.cat(Ss)

    # ---------------- release ----------------
    @staticmethod
    def _cut(spec):
        """axis, cut-tip coordinates and distance (along the axis) to the still-bonded strip of the plate."""
        a = spec.get("axis", 0)
        if "keep" in spec:
            lo, hi = map(float, spec["keep"])
            return a, [lo, hi], lambda X: torch.relu(lo - X[:, a:a + 1]) + torch.relu(X[:, a:a + 1] - hi)
        p, sgn = float(spec["pos"]), (1.0 if spec.get("from", "min") == "min" else -1.0)
        return a, [p], lambda X: torch.relu(sgn * (p - X[:, a:a + 1]))

    def _phi(self, spec):
        """Distance function of the hard boundary condition (Eq. 7, right)."""
        if spec["type"] == "full":
            return lambda X: torch.ones_like(X[:, :1])
        _, _, dist = self._cut(spec)
        return lambda X: torch.sqrt((X[:, 2:3] - self.g.z0) ** 2 + dist(X) ** 2 + 1e-6) / self.g.H

    def _rfeat(self, spec, X):
        """Release-stage input features (Section 2.4): interface ramps and cut-tip features."""
        H = self.g.H
        if not self.s.rfeat:
            return None
        f = [torch.relu(X[:, a:a + 1] - c) / (H / 4) for a, c in self.g.interface_planes()]
        if spec["type"] == "cut":
            a, tips, dist = self._cut(spec)
            f.append(dist(X) / H)
            for p in tips:
                r = torch.sqrt((X[:, a:a + 1] - p) ** 2 + (X[:, 2:3] - self.g.z0) ** 2 + 1e-8)
                f.append(torch.sqrt(r / H))
        return torch.cat(f, 1) if f else None

    def _rnet(self, spec):
        f = self._rfeat(spec, torch.zeros(1, 3, device=self.dev))
        return self._net(0 if f is None else f.shape[1])

    def _u_rel(self, net, spec, X):
        return self.U_rel * self._phi(spec)(X) * net(X, self._rfeat(spec, X))

    def _get_cloud(self):
        if self._cloud is None:
            Xc, wc = self.g.sample(self.s.n_cloud, None, self.dev, self.s.stratified)
            _, S0 = self.asbuilt(Xc)
            self._cloud = (Xc, wc, S0, self.g.is_support(Xc))
        return self._cloud

    def _refine_tip(self, a, pos, Xc, wc, S0, stiffc):
        """Replace the points within tip_r of a cut tip by tip_n denser points with the same total weight."""
        R = self.s.tip_r
        lo, hi = self.g.lo.clone(), self.g.hi.clone()
        lo[a], hi[a] = max(float(lo[a]), pos - R), min(float(hi[a]), pos + R)
        hi[2] = min(float(hi[2]), self.g.z0 + R)
        lo, hi = lo.to(self.dev), hi.to(self.dev)
        inb = ((Xc >= lo) & (Xc <= hi)).all(1)
        V = float(wc[inb].sum())
        if V <= 0:
            return Xc, wc, S0, stiffc
        kept = []
        while sum(k.shape[0] for k in kept) < self.s.tip_n:
            C = lo + torch.rand(4 * self.s.tip_n, 3, device=self.dev) * (hi - lo)
            kept.append(C[self.g.inside(C)])
        Xt = torch.cat(kept)[:self.s.tip_n]
        _, St = self.asbuilt(Xt)
        return (torch.cat([Xc[~inb], Xt]), torch.cat([wc[~inb], torch.full((Xt.shape[0],), V / Xt.shape[0], device=self.dev)]),
                torch.cat([S0[~inb], St]), torch.cat([stiffc[~inb], self.g.is_support(Xt)]))

    def train_release(self, name, spec):
        s = self.s
        Xc, wc, S0, stiffc = self._get_cloud()
        if spec["type"] == "cut" and s.tip_n:
            a, tips, _ = self._cut(spec)
            for pos in tips:
                Xc, wc, S0, stiffc = self._refine_tip(a, pos, Xc, wc, S0, stiffc)
        net = self._rnet(spec)
        opt = torch.optim.Adam(net.parameters(), lr=s.lr_release)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s.iters_release)
        Xbar = (wc[:, None] * Xc).sum(0) / wc.sum()
        hist, t0 = [], time.time()

        def objective(idx):                                   # Eq. 4 (+ rigid-body penalty after removal)
            X = (Xc if idx is None else Xc[idx]).clone().requires_grad_(True)
            w, S, st_ = (wc, S0, stiffc) if idx is None else (wc[idx], S0[idx], stiffc[idx])
            u = self._u_rel(net, spec, X)
            e = sym(grad_u(u, X)) / EPS_UNIT
            dens = 0.5 * (e * self._stress(e, st_)).sum((1, 2)) + (S * e).sum((1, 2))
            loss = (w * dens).sum() / w.sum()
            if spec["type"] == "full":
                um = (w[:, None] * u).sum(0) / w.sum()
                rot = (w[:, None] * torch.cross(X - Xbar, u, dim=1)).sum(0) / w.sum()
                loss = loss + s.rb_weight * (((um / self.U_rel) ** 2).sum() + ((rot / (self.U_rel * self.g.Lmax)) ** 2).sum())
            return loss

        for _ in range(s.iters_release):                      # Adam warm start on mini-batches
            idx = torch.randint(0, Xc.shape[0], (s.batch,), device=self.dev)
            loss = objective(idx)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            hist.append(loss.item())
        if s.lbfgs_release:                                   # full-batch L-BFGS on the fixed cloud
            lb = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
            for _ in range(max(1, s.lbfgs_release // 50)):
                def clo():
                    lb.zero_grad(); L = objective(None); L.backward(); return L
                hist.append(float(lb.step(clo).detach()))
        self.history[f"release_{name}"] = hist
        self.log(f"[{self.g.name}] release {name:10s} energy {hist[-1]:.3e}  ({time.time()-t0:.0f}s)")
        self.release_nets[name], self.specs[name] = self._freeze(net), spec

    def train(self, scenarios):
        torch.manual_seed(self.s.seed); np.random.seed(self.s.seed)
        t0 = time.time()
        self.train_build()
        for name, spec in scenarios.items():
            if spec["type"] == "onplate":
                self.specs[name] = spec
            else:
                self.train_release(name, spec)
        self.train_time = time.time() - t0
        self.log(f"[{self.g.name}] training done in {self.train_time/60:.1f} min")
        return self

    # ---------------- evaluation at a real strain eps ----------------
    def _t(self, X):
        return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.dev)

    def displacement(self, scenario, X, eps, kind="total"):
        """Displacement (mm) for inherent strain eps. kind='release': change caused by the cut/removal only
        (what a cut test measures); kind='total': as-built + release."""
        X, a = self._t(X), eps / EPS_UNIT
        u_ab, _ = self.asbuilt(X)
        spec = self.specs[scenario]
        if spec["type"] == "onplate":
            return (a * u_ab).cpu().numpy()
        with torch.no_grad():
            u_r = self._u_rel(self.release_nets[scenario], spec, X)
        return (a * (u_r if kind == "release" else u_ab + u_r)).cpu().numpy()

    def stress(self, scenario, X, eps, bs=4000):
        """Residual stress tensor (MPa), shape (N,3,3), for inherent strain eps."""
        X, a, out = self._t(X), eps / EPS_UNIT, []
        _, S0 = self.asbuilt(X)
        spec = self.specs[scenario]
        if spec["type"] == "onplate":
            return (a * S0 * self.MPa).cpu().numpy()
        for i in range(0, X.shape[0], bs):
            Xb = X[i:i + bs].clone().requires_grad_(True)
            e = sym(grad_u(self._u_rel(self.release_nets[scenario], spec, Xb), Xb)) / EPS_UNIT
            out.append((S0[i:i + bs] + self._stress(e, self.g.is_support(Xb))).detach())
        return (a * torch.cat(out) * self.MPa).cpu().numpy()

    # ---------------- persistence ----------------
    def save(self, path):
        torch.save({"settings": asdict(self.s), "material": asdict(self.m), "geometry": self.g.describe(),
                    "zk": self.zk, "specs": self.specs,
                    "build": [n.state_dict() for n in self.build_nets],
                    "release": {k: n.state_dict() for k, n in self.release_nets.items()}}, path)

    @staticmethod
    def load(path, geom, device=None, log=print):
        d = torch.load(path, map_location="cpu", weights_only=False)
        known = {f.name for f in fields(SolverSettings)}
        st = SolverSettings(**{k: v for k, v in d["settings"].items() if k in known})
        obj = ISPINN(geom, Material(**d["material"]), st, device, log)
        obj.zk, obj.specs = d["zk"], d["specs"]
        for sd in d["build"]:
            n = obj._net(1); n.load_state_dict(sd); obj.build_nets.append(obj._freeze(n))
        for k, sd in d["release"].items():
            n = obj._rnet(obj.specs[k]); n.load_state_dict(sd); obj.release_nets[k] = obj._freeze(n)
        return obj
