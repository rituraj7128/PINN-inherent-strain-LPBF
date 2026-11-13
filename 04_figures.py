"""
Step 4 — paper figures (300 dpi PNG, OriginPro-like style) from the results of steps 2 and 3.

    python 04_figures.py            # all figures -> figures/
    python 04_figures.py 4 6        # only Fig. 4 and Fig. 6
Set ORIGIN=0 for the plain matplotlib style.
Fig. 3 (ablation) uses results/ablation_development.json: ratios measured during development with
intermediate versions of the solver (A0-A3, see README); A4 is the final solver of this package.
"""
import os, json, glob
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import ispinn as isp

ORIGIN = os.environ.get("ORIGIN", "1") == "1"
FD = "figures"
os.makedirs(FD, exist_ok=True)
INK, MUT, PINN, SIM, FE, GRID = "#1a1a19", "#6b6a66", "#2a78d6", "#eb6834", "#6b6a66", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUT, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "lines.markersize": 4.5, "savefig.dpi": 300, "savefig.bbox": "tight"})
W1, W2 = 3.54, 7.48     # single / double column width (inch)
X8 = 2 + 9.7 * np.arange(8)
EXP = {"C": [3.29, 2.12, 1.25, 0.65, 0.25, -0.05, -0.15, -0.05], "D": [2.34, 1.47, 0.82, 0.35, 0.05, -0.12, -0.03, -0.05]}
CT = {"C": [1.171194, 0.650631, 0.291737, 0.163488, 0.042567, 0.305136, 0.576517, 1.163140],
      "D": [1.013625, 0.415781, 0.231344, 0.161453, 0.092261, 0.203862, 0.413189, 1.021384]}
RS = {"C": [525, 651, 730, 701, 760, 748, 768, 562], "D": [284, 468, 537, 544, 502, 622, 550, 409]}
SIMpt0 = {"C": 2.94496, "D": 2.54135}
_FEF = "results/fe_cantilever_1.0x1.0x0.2812.json"
_fe = json.load(open(_FEF)) if os.path.exists(_FEF) else {"cut_C": [], "cut_D": [], "removed_detr": []}
F = {"cut_C": _fe["cut_C"], "cut_D": _fe["cut_D"], "removed": _fe["removed_detr"]}
RR = [json.load(open(f)) for f in sorted(glob.glob("results/seed*.json"))]
R0 = dict(RR[0]); R0["simufact_twin"] = json.load(open("data/simufact_twin_cantilever.json"))
_quiet = lambda *a: None
_MODEL = isp.ISPINN.load("models/cantilever_seed0.pt", isp.paper_cantilever(), device="cpu", log=_quiet)
def _unit():   # PINN at unit strain (seed 0), for the FE comparison figures
    g = isp.paper_cantilever(); P = isp.top_line(g, X8)
    return {"pinn_C": _MODEL.displacement("cut_C", P, isp.EPS_UNIT, "release")[:, 2].tolist(),
            "pinn_D": _MODEL.displacement("cut_D", P, isp.EPS_UNIT, "release")[:, 2].tolist(),
            "pinn_removed": isp.detrend(_MODEL.displacement("removed", P, isp.EPS_UNIT)[:, 2], X8).tolist()}
def _prof(model=None):
    model = model or _MODEL
    z = np.linspace(6.42, 8.98, 60)
    _, S = model.asbuilt(torch.tensor(np.c_[0 * z + 30.25, 0 * z + 0.5, z], dtype=torch.float32))
    return {"z": z.tolist(), "sxx": (S[:, 0, 0].numpy() * model.MPa).tolist()}
tag = lambda a, t: a.text(-0.20 if ORIGIN else -0.13, 1.04, t, transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")
if ORIGIN:   # OriginPro look: Arial-type font, closed frame, inward major+minor ticks on all sides, no grid
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8.5, "axes.labelsize": 9, "axes.labelweight": "bold",
                         "axes.titlesize": 9, "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black",
                         "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 5, "ytick.major.size": 5,
                         "xtick.minor.size": 2.5, "ytick.minor.size": 2.5, "xtick.color": "black", "ytick.color": "black",
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.markersize": 6,
                         "legend.frameon": False})
def grid(a):
    if ORIGIN:
        return
    a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)

# ---------------------------------------------------------------- Fig. 1 schematic
def fig1():
    fig = plt.figure(figsize=(W2, 4.4))
    a = fig.add_axes([0.04, 0.62, 0.44, 0.34]); b = fig.add_axes([0.54, 0.62, 0.44, 0.34]); c = fig.add_axes([0.02, 0.0, 0.96, 0.58])
    def part(ax, solids, teeth_ranges, cuts, probes, H, ht, nl):
        ax.add_patch(Rectangle((-4, -2.0), 80, 2.0, fc="#efeeea", ec="#9a9893", lw=0.5, hatch="////"))
        ax.add_patch(Rectangle((-4, -3.2), 80, 1.2, color="#d9d8d4", lw=0))
        for (x0, x1) in teeth_ranges:
            for xt in np.arange(x0 + 0.5, x1 - 0.5, 2.0):
                ax.add_patch(Rectangle((xt, 0), 1.0, ht, color="#9fb9d9", lw=0))
        for (x0, z0, x1, z1) in solids:
            ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, color="#2f5d8f", lw=0))
        for z in np.arange(1, nl) * H / nl:
            ax.plot([-1, 73], [z, z], color="white", lw=0.25, alpha=0.6)
        for (x0, x1) in cuts:
            ax.plot([x0, x1], [0, 0], color=SIM, lw=2.2, solid_capstyle="butt")
        ax.plot(probes, [H + 0.4] * len(probes), "v", color=INK, ms=3.5)
        ax.set_xlim(-4, 76); ax.set_ylim(-9.5, 14.5); ax.set_aspect("equal"); ax.axis("off")
    nl = round(9 / 0.55)
    part(a, [(0, 6.4, 72, 9), (62, 0, 72, 6.4), (0, 0, 3, 6.4)], [(3, 62)], [(0, 60)], X8, 9.0, 6.4, nl)
    a.annotate("", (0, -4.6), (60, -4.6), arrowprops=dict(arrowstyle="<->", lw=0.6, color=SIM)); a.text(30, -6.3, "EDM cut at z = 0: 60 mm (C) / 54 mm (D); hatched: 2 mm extrusion, rigid", ha="center", fontsize=6.3, color=INK)
    a.text(67, 3.2, "fixed\nblock", ha="center", va="center", color="white", fontsize=6)
    a.text(36, 7.7, "solid beam 2.6 mm", ha="center", va="center", color="white", fontsize=6.5)
    a.text(32, 3.2, "comb teeth 6.4 mm\n(homogenised layer)", ha="center", va="center", fontsize=6.5, color=INK)
    a.text(0, -9.3, f"72 × 12 × 9 mm; {nl} lumped layers of {9 / nl:.4f} mm; ▼ measurement points", fontsize=6.3, color=MUT)
    tag(a, "(a)"); a.text(0, 11.6, "Cantilevers C and D (Dinesh & Sahu, 2025)", fontsize=7.5)
    part(b, [(0, 8.4, 72, 11), (31, 0, 41, 8.4), (0, 0, 3, 8.4), (69, 0, 72, 8.4)], [(3, 31), (41, 69)], [(0, 29.0), (41.85, 72)],
         np.array([1.1, 10.5, 19.9, 26.5, 35.2, 43.9, 51.2, 61.3, 70.3]), 11.0, 8.4, 20)
    b.text(36, 4.2, "pillar", ha="center", va="center", color="white", fontsize=6, rotation=90)
    b.text(0, -5.6, "cut on both sides (orange); bond kept for 29.0 ≤ x ≤ 41.85 mm", fontsize=6.3, color=MUT); b.text(0, -9.3, "72 × 12 × 11 mm; teeth 8.4 mm; ▼ Simufact query points", fontsize=6.3, color=MUT)
    tag(b, "(b)"); b.text(0, 13.2, "Twin cantilever (taller supports; Simufact Additive)", fontsize=7.5)
    # workflow
    c.axis("off"); c.set_xlim(0, 100); c.set_ylim(0, 40)
    def box(x, y, w, h, txt, fc):
        c.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=MUT, lw=0.6))
        c.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.6)
    def arr(x0, y0, x1, y1):
        c.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK))
    box(1, 24, 20, 13, "Build, step k = 1…N\nlayer k gets ε$^{in}$ = −ε$_0$ I\nmin Π$_k$(Δu$_k$), network N$_k$\n+ kink feature max(0, z−z$_{k-1}$)/h", "#e7f1fb")
    box(27, 24, 18, 13, "As-built state\nu$_0$ = Σ Δu$_k$\nσ$_0$ (accumulated)", "#f0efec")
    box(51, 30, 22, 8, "Partial cut (C, D) / two-sided cut\nmin Π$_r$, distance-function BC\n+ interface & √r tip features", "#e7f1fb")
    box(51, 19, 22, 8, "Removal from plate\nmin Π$_r$ + rigid-body penalty", "#e7f1fb")
    box(79, 24, 20, 13, "Unit-strain fields\nu$_0$, σ$_0$ (ε$_0$ = 10$^{-3}$)\nlinear in ε*", "#f0efec")
    arr(21, 30.5, 27, 30.5); arr(45, 32, 51, 34); arr(45, 28, 51, 23); arr(73, 34, 79, 32); arr(73, 23, 79, 28)
    box(63, 3, 36, 12, "Calibration (one number)\nε* = ε$_0$ (u$_0$·u$_{meas}$)/(u$_0$·u$_0$)\non the measured cut profile of C", "#f9eeda")
    box(15, 3, 40, 12, "Prediction with the same ε*: D, removal shape,\nstresses, new geometries\n(each geometry retrained from physics only)", "#e5f4e4")
    arr(89, 24, 89, 15.6); arr(63, 9, 55.5, 9)
    c.text(1, 39.5, "Energy-based PINN, no simulation or measurement data in the loss; stratified quadrature, Adam → full-batch L-BFGS", fontsize=7)
    tag(c, "(c)"); c.text(-1.5, 41.5, "", fontsize=1)
    fig.savefig(f"{FD}/fig1_model.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2 ablation + through-thickness stress
def fig3():
    A = json.load(open("results/ablation_v2.json"))
    AS = {k: v for k, v in (json.load(open("results/ablation_seeds.json")) if os.path.exists("results/ablation_seeds.json") else {}).items() if v["n"] >= 3}
    items = [("A0", "A0"), ("A0+stratified", "+strat."), ("A0+lbfgs", "+L-BFGS"), ("A0+layer_enrichment", "+layer"), ("A0+release_features", "+rel. feat."),
             ("A0+tip_refinement", "+tip"), ("full", "full"), ("full-stratified", "−strat."), ("full-lbfgs", "−L-BFGS"), ("full-layer_enrichment", "−layer"),
             ("full-release_features", "−rel. feat."), ("full-tip_refinement", "−tip")]
    items = [it for it in items if it[0] in A]; n = len(items); nadd = 6
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})
    a = ax[0]; keys = [("cut_C", "cut C"), ("cut_D", "cut D"), ("bow", "removal bow"), ("vm_on", "on-plate stress")]
    cols = ["#2a78d6", "#1f9e89", "#eb6834", "#8a63d2"]; w = 0.2
    for i, (k, nm) in enumerate(keys):
        val = [AS[it]["mean"][k] if it in AS else A[it][k] for it, _ in items]
        err = [AS[it]["sd"][k] if it in AS else np.nan for it, _ in items]
        a.bar(np.arange(n) + (i - 1.5) * w, val, w * 0.9, color=cols[i], label=nm, yerr=err if AS else None,
              error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK))
    a.axhline(1, color=INK, lw=0.7, ls="--"); a.set_xticks(range(n))
    a.set_xticklabels([l + (" *" if it in AS else "") for it, l in items], rotation=35, ha="right", fontsize=7.5)
    a.axvline(nadd - 0.5, color=MUT, lw=0.8); a.set_ylim(0, 1.3); a.set_ylabel("PINN / FE (unit strain)")
    a.text((nadd - 1) / 2, 1.2, "one feature added to A0", ha="center", fontsize=7.2, color=MUT)
    a.text(nadd + (n - nadd - 1) / 2, 1.2, "one feature removed from full", ha="center", fontsize=7.2, color=MUT)
    a.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=7.2); grid(a); tag(a, "(a)")
    b = ax[1]; g = isp.paper_cantilever()
    z = np.array(_fe["profile_z"]); sx = np.array(_fe["profile_sxx"]); m = z > 6.4
    b.plot(sx[m], z[m], "s", color=FE, ms=4, label="FE (element mean)")
    for f, lb, cl, ls in (("models/ablation/A0.pt", "A0 standard", MUT, ":"), ("models/ablation/A0+layer_enrichment.pt", "A0 + layer enrichment", "#1f9e89", "--"),
                          ("models/cantilever_seed0.pt", "full solver", PINN, "-")):
        if os.path.exists(f):
            pr = _prof(isp.ISPINN.load(f, g, device="cpu", log=_quiet)); b.plot(pr["sxx"], pr["z"], ls, color=cl, label=lb)
    b.set_xlabel("σ$_{xx}$ per 0.1 % inherent strain (MPa)"); b.set_ylabel("z (mm)"); b.set_ylim(6.4, 9.0); grid(b)
    b.legend(frameon=False, loc="upper left", fontsize=6.5); tag(b, "(b)"); b.set_title("as built, beam section at x = 30 mm", loc="left")
    plt.tight_layout(); fig.savefig(f"{FD}/fig3_ablation.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 FE vs PINN unit strain
def fig2():
    d = _unit(); f05 = json.load(open("results/fe_cantilever_0.5x1.0x0.1406.json")) if os.path.exists("results/fe_cantilever_0.5x1.0x0.1406.json") else None
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    for i, c in enumerate("CD"):
        a = ax[i]; a.plot(X8, F[f"cut_{c}"], "s", color=FE, label="FE, 1×1×0.281 mm")
        if f05: a.plot(X8, f05[f"cut_{c}"], "D", mfc="none", color=FE, label="FE, 0.5×1×0.141 mm")
        a.plot(X8, d[f"pinn_{c}"], "-o", color=PINN, ms=3, label="PINN")
        a.set_title(f"release after cut {c}", loc="left"); a.set_xlabel("x from free end (mm)"); grid(a); tag(a, f"({'ab'[i]})")
    ax[0].set_ylabel("Δz per 0.1 % strain (mm)"); ax[0].legend(frameon=False)
    a = ax[2]; X = X8
    a.plot(X, isp.detrend(np.array(F["removed"]), X), "s", color=FE, label="FE"); a.plot(X, d["pinn_removed"], "-o", color=PINN, ms=3, label="PINN")
    a.set_title("shape after removal (detrended)", loc="left"); a.set_xlabel("x from free end (mm)"); grid(a); tag(a, "(c)")
    plt.tight_layout(); fig.savefig(f"{FD}/fig2_fe_verification.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 calibration, prediction, removal
def band(key):
    A = np.array([r[key] for r in RR]); return A.mean(0), A.min(0), A.max(0)
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5))
    for i, c in enumerate("CD"):
        a = ax[i]; m, lo, hi = band(f"cut_{c}")
        a.fill_between(X8, lo, hi, color=PINN, alpha=0.18, lw=0)
        a.plot(X8, EXP[c], "s--", color=INK, lw=0.9, label="experiment")
        a.plot(X8, m, "-o", color=PINN, ms=3, label=f"PINN (mean of {len(RR)} seeds)")
        a.plot(X8[0], SIMpt0[c], "D", color=SIM, ms=5, label="Simufact (source paper)")
        a.set_title(f"cantilever {c}: {'calibration' if c == 'C' else 'prediction'}", loc="left"); a.set_xlabel("x from free end (mm)"); grid(a); tag(a, f"({'ab'[i]})")
    ax[0].set_ylabel("Z after cut (mm)"); ax[0].legend(frameon=False)
    a = ax[2]; m, lo, hi = band("removed_detr")
    for c, mk, al in (("C", "s", 1), ("D", "^", 0.55)):
        a.plot(X8, isp.detrend(np.array(CT[c]), X8), mk + "--", color=INK, alpha=al, lw=0.9, label=f"CT, {c}")
    a.fill_between(X8, lo, hi, color=PINN, alpha=0.18, lw=0); a.plot(X8, m, "-o", color=PINN, ms=3, label="PINN")
    a.set_title("shape after removal (prediction)", loc="left"); a.set_ylabel("deviation (mm)"); a.set_xlabel("x from free end (mm)")
    grid(a); a.legend(frameon=False, loc="upper center"); tag(a, "(c)")
    plt.tight_layout(); fig.savefig(f"{FD}/fig4_calibration_prediction.png"); plt.close(fig)

# ---------------------------------------------------------------- field maps (mid-plane y = 0)
def fields(model, geom, scen, eps, xs, zs):
    Xg, Zg = np.meshgrid(xs, zs); P = np.c_[Xg.ravel(), 0 * Xg.ravel(), Zg.ravel()]
    ins = geom.inside(torch.tensor(P, dtype=torch.float32)).numpy(); sup = geom.is_support(torch.tensor(P, dtype=torch.float32)).numpy()
    u = model.displacement(scen, P, eps); S = model.stress(scen, P, eps); vm = isp.von_mises(S)
    vm = np.where(ins & ~sup, vm, np.nan); uz = np.where(ins, u[:, 2], np.nan)
    return Xg, Zg, uz.reshape(Xg.shape), vm.reshape(Xg.shape), (ins & sup).reshape(Xg.shape)
def fig5():
    eps = R0["eps"]; g = isp.paper_cantilever(); s = _MODEL
    xs, zs = np.linspace(0.05, 71.95, 290), np.linspace(0.05, 8.95, 50)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 4.6))
    for i, (sc, t) in enumerate((("onplate", "on plate (as built)"), ("cut_C", "after partial cut C"))):
        Xg, Zg, uz, vm, sup = fields(s, g, sc, eps, xs, zs)
        a = ax[i]; im = a.pcolormesh(Xg, Zg, vm, cmap="YlOrRd", vmin=0, vmax=1500, shading="auto")
        a.contourf(Xg, Zg, sup.astype(float), levels=[0.5, 1.5], colors=["#d9d8d4"])
        a.text(32, 3.2, "homogenised comb teeth (stress not shown)", ha="center", va="center", fontsize=6.3, color=MUT)
        a.set_aspect("equal"); a.set_title(f"von Mises stress, {t}", loc="left"); a.set_ylabel("z (mm)"); tag(a, f"({'ab'[i]})")
    fig.colorbar(im, ax=ax[:2], label="MPa", shrink=0.8, pad=0.02)
    a = ax[2]; sx = np.mean([r["sxx_on"] for r in RR], 0); sy = np.mean([r["syy_on"] for r in RR], 0)
    a.plot(X8, RS["C"], "s--", color=INK, lw=0.9, label=f"XRD, C (mean {np.mean(RS['C']):.0f})")
    a.plot(X8, RS["D"], "^--", color=INK, alpha=0.55, lw=0.9, label=f"XRD, D (mean {np.mean(RS['D']):.0f})")
    a.plot(X8, sx, "-o", color=PINN, ms=3, label=f"PINN σ$_{{xx}}$ (mean {np.mean(sx):.0f})")
    a.plot(X8, sy, "--o", color=PINN, ms=3, mfc="none", label=f"PINN σ$_{{yy}}$ (mean {np.mean(sy):.0f})"); a.axhline(800, color=MUT, lw=0.7, ls=":")
    a.text(1, 815, "as-built yield, XY (EOS)", ha="left", fontsize=6, color=MUT)
    a.set_ylim(0, 2150); a.set_xlabel("x from free end (mm)"); a.set_ylabel("stress, top (MPa)"); grid(a)
    a.legend(frameon=False, ncol=2, loc="upper center", fontsize=6); tag(a, "(c)")
    fig.savefig(f"{FD}/fig5_stress.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 6 twin cantilever
def fig6():
    S = R0["simufact_twin"]; eps = R0["eps"]
    gt = isp.twin_cantilever()
    fig = plt.figure(figsize=(W2, 4.0))
    a = fig.add_axes([0.07, 0.56, 0.40, 0.38]); b = fig.add_axes([0.57, 0.56, 0.40, 0.38]); c = fig.add_axes([0.07, 0.06, 0.82, 0.36])
    m = np.mean([r["twin_dz_cut"] for r in RR], 0); lo = np.min([r["twin_dz_cut"] for r in RR], 0); hi = np.max([r["twin_dz_cut"] for r in RR], 0)
    a.fill_between(R0["twin_x"], lo, hi, color=PINN, alpha=0.18, lw=0); a.plot(R0["twin_x"], m, "-", color=PINN, label="PINN (ε* of C)")
    a.plot(S["dz_x"], S["dz"], "s", color=SIM, label="Simufact Additive"); a.set_xlabel("x (mm)"); a.set_ylabel("Z after cut (mm)"); grid(a); a.legend(frameon=False, loc="upper center"); tag(a, "(a)")
    for xe, i in ((S["dz_x"][0], 0), (S["dz_x"][-1], -1)):
        e = 100 * (np.mean([r["twin_dz_at_sim"][i] for r in RR]) - S["dz"][i]) / S["dz"][i]
        a.annotate(f"{e:+.0f} %", (xe, S["dz"][i]), (0, -14), textcoords="offset points", ha="center", fontsize=7, color=PINN)
    a.text(0.5, 0.70, "teeth 8.4 mm (calibration: 6.4 mm)", transform=a.transAxes, ha="center", fontsize=6.8, color=MUT)
    on = np.mean([r["twin_vm_on"] for r in RR], 0); ct = np.mean([r["twin_vm_cut"] for r in RR], 0)
    o1, o2 = np.argsort(S["vm_on_x"]), np.argsort(S["vm_cut_x"])
    b.plot(np.array(S["vm_on_x"])[o1], np.array(S["vm_on"])[o1], "s", color=SIM, label="Simufact, on plate")
    b.plot(np.array(S["vm_on_x"])[o1], on[o1], "-o", color=PINN, ms=3, label="PINN, on plate")
    b.plot(np.array(S["vm_cut_x"])[o2], np.array(S["vm_cut"])[o2], "^", color=SIM, mfc="none", label="Simufact, after cut")
    b.plot(np.array(S["vm_cut_x"])[o2], ct[o2], "--o", color=PINN, ms=3, mfc="none", label="PINN, after cut")
    b.set_ylim(0, 1500); b.set_xlabel("x (mm)"); b.set_ylabel("von Mises, top (MPa)"); grid(b); b.legend(frameon=False, ncol=2, fontsize=6, loc="lower center"); tag(b, "(b)")
    if True:
        t = isp.ISPINN.load("models/twin_seed0.pt", gt, device="cpu", log=_quiet)
        xs, zs = np.linspace(0.05, 71.95, 290), np.linspace(0.05, 10.95, 60)
        Xg, Zg, uz, vm, sup = fields(t, gt, "cut", eps, xs, zs)
        im = c.pcolormesh(Xg, Zg, uz, cmap="viridis", shading="auto"); c.set_aspect("equal")
        fig.colorbar(im, ax=c, label="Z displacement (mm)", shrink=0.9, pad=0.01)
        c.set_title("PINN Z displacement after cutting both arms (mid-plane)", loc="left"); c.set_xlabel("x (mm)"); c.set_ylabel("z (mm)"); tag(c, "(c)")
    fig.savefig(f"{FD}/fig6_twin.png"); plt.close(fig)

if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k in todo:
        globals()[f"fig{k}"](); print("fig", k, "done", flush=True)
