"""
Step 3 — verify the PINN against an independent finite-element solution of the SAME model (Section 4).

    python 03_fe_verification.py                     # reference mesh 1 x 1 x 0.275 mm (~20 min on a laptop CPU)
    python 03_fe_verification.py --refined           # + meshes 0.5 x 1 x 0.275 and 1 x 1 x 0.1375 (~1.5 h more)

Everything is per unit inherent strain (1e-3). Outputs in results/:
  fe_cantilever_<mesh>.json  FE cut C/D release, bow, top-surface stress, through-thickness stress at x = 30
  pinn_vs_fe.json            Table 2 of the paper: PINN/FE ratios and the build/release error split
  fe_twin.json               FE of the twin cantilever (PINN/FE end deflection, Section 7)
Needs: pip install scikit-fem pyamg
"""
import argparse, json, os
import numpy as np, torch
import ispinn as isp
import fe_reference as fe

ap = argparse.ArgumentParser()
ap.add_argument("--refined", action="store_true")
ap.add_argument("--finest", action="store_true", help="also the mesh refined in both directions")
ap.add_argument("--model", default="models/cantilever_seed0.pt")
a = ap.parse_args()
os.makedirs("results", exist_ok=True)
m = isp.Material(); X8 = 2 + 9.7 * np.arange(8)
g = isp.paper_cantilever(); Csup = g.support.voigt(m.E, m.nu)
quiet = lambda *x: None

def lh(geom):                                     # same lumped layers as the PINN: round(H / 0.55) layers
    return geom.H / max(1, round(geom.H / 0.55))

def fe_run(geom, h, cuts, probe, S_func=None, fields=False):
    return fe.run_fe(fe.boxes_from_geometry(geom), layer_h=lh(geom), E=m.E, nu=m.nu, h=h, cuts=cuts, probe_x=probe,
                     support=geom.support.voigt(m.E, m.nu), log=quiet, return_fields=fields, S_func=S_func)

def summarise(r, h):
    c, S = r["centroids"], r["S_el"]
    sel = (np.abs(c[0] - 30.25) < h[0] / 2 + 1e-6) & (np.abs(c[1] - 0.5) < 0.51) & (c[2] > g.H - 2.6)
    o = np.argsort(c[2, sel])
    return {"mesh_mm": list(h), "dofs": int(r["dofs"]), "time_s": float(r["time_s"]),
            "cut_C": np.asarray(r["cut_C"]).tolist(), "cut_D": np.asarray(r["cut_D"]).tolist(),
            "removed_detr": isp.detrend(np.asarray(r["removed"]), X8).tolist(),
            "vm_top_onplate": np.asarray(r["asbuilt_vm_top"]).tolist(), "vm_top_cut_C": np.asarray(r["vm_top_cut_C"]).tolist(),
            "profile_z": c[2, sel][o].tolist(), "profile_sxx": S[0, 0, sel][o].tolist()}

hz = lh(g) / 2                                    # two elements per lumped layer
meshes = [(1.0, 1.0, hz)] + ([(0.5, 1.0, hz), (1.0, 1.0, hz / 2)] if a.refined else []) + ([(0.5, 1.0, hz / 2)] if a.finest else [])
FE = {}
for h in meshes:
    fn = f"results/fe_cantilever_{h[0]}x{h[1]}x{h[2]:.4g}.json"
    if os.path.exists(fn) and os.path.getmtime(fn) > os.path.getmtime("fe_reference.py"):
        FE[h] = json.load(open(fn)); print("FE cantilever, mesh", h, "(stored)", flush=True); continue
    print("FE cantilever, mesh", h, flush=True)
    FE[h] = summarise(fe_run(g, h, {"C": 60.0, "D": 54.0}, X8, fields=True), h)
    json.dump(FE[h], open(fn, "w"), indent=1)
    print(f"   cut C {FE[h]['cut_C'][0]:.4f}  cut D {FE[h]['cut_D'][0]:.4f}  bow {np.ptp(FE[h]['removed_detr']):.4f}"
          f"  vm on plate {np.mean(FE[h]['vm_top_onplate']):.1f}  ({FE[h]['time_s']:.0f} s)", flush=True)
ref = FE[meshes[0]]

# ---- PINN vs FE (Table 2) and build/release split
s = isp.ISPINN.load(a.model, g, device="cpu", log=quiet)
P = isp.top_line(g, X8); top = np.c_[X8, 0 * X8, 0 * X8 + g.H - 0.14]
pinn = {k: s.displacement(f"cut_{k}", P, isp.EPS_UNIT, "release")[:, 2] for k in "CD"}
pinn_bow = float(np.ptp(isp.detrend(s.displacement("removed", P, isp.EPS_UNIT)[:, 2], X8)))
vm_on = float(np.mean(isp.von_mises(s.stress("onplate", top, isp.EPS_UNIT))))
vm_cut = float(np.mean(isp.von_mises(s.stress("cut_C", top, isp.EPS_UNIT))))
def S_pinn(Q):                                    # PINN as-built stress (MPa) at the FE quadrature points
    out = []
    for i in range(0, len(Q), 20000):
        _, S = s.asbuilt(torch.tensor(Q[i:i + 20000], dtype=torch.float32)); out.append(S.numpy() * s.MPa)
    return np.concatenate(out)
print("FE release driven by the PINN as-built stress", flush=True)
mix = fe_run(g, meshes[0], {"C": 60.0, "D": 54.0}, X8, S_func=S_pinn)
res = {"model": a.model}
for k in "CD":
    res[f"cut_{k}"] = {"fe": ref[f"cut_{k}"][0], "pinn": float(pinn[k][0]), "ratio": float(pinn[k][0] / ref[f"cut_{k}"][0]),
                       "build_stage_ratio": float(mix[f"cut_{k}"][0] / ref[f"cut_{k}"][0]),
                       "release_stage_ratio": float(pinn[k][0] / mix[f"cut_{k}"][0])}
fb = float(np.ptp(ref["removed_detr"]))
res["bow"] = {"fe": fb, "pinn": pinn_bow, "ratio": pinn_bow / fb}
res["vm_on"] = {"fe": float(np.mean(ref["vm_top_onplate"])), "pinn": vm_on, "ratio": vm_on / float(np.mean(ref["vm_top_onplate"]))}
res["vm_cut_C"] = {"fe": float(np.mean(ref["vm_top_cut_C"])), "pinn": vm_cut, "ratio": vm_cut / float(np.mean(ref["vm_top_cut_C"]))}
json.dump(res, open("results/pinn_vs_fe.json", "w"), indent=1)
print("\nPINN / FE at unit strain (Table 2):")
for k in ("cut_C", "cut_D", "bow", "vm_on", "vm_cut_C"):
    extra = f"   build {res[k]['build_stage_ratio']:.3f} x release {res[k]['release_stage_ratio']:.3f}" if k.startswith("cut") else ""
    print(f"  {k:9s} FE {res[k]['fe']:8.3f}  PINN {res[k]['pinn']:8.3f}  ratio {res[k]['ratio']:.3f}{extra}")

# ---- twin cantilever FE (end deflections, Section 7)
gt = isp.twin_cantilever(); xt = np.linspace(0, 72, 37)
print("\nFE twin cantilever", flush=True)
rt = fe_run(gt, (1.0, 1.0, lh(gt) / 2), {"both": tuple(isp.SCEN_TWIN["cut"]["keep"])}, xt)
fe_tot = np.asarray(rt["cut_both"]) + np.asarray(rt["asbuilt_dz"])
ratios = []
for f in sorted(__import__("glob").glob("models/twin_seed*.pt")):
    t = isp.ISPINN.load(f, gt, device="cpu", log=quiet)
    pt = t.displacement("cut", isp.top_line(gt, [0.0, 72.0]), isp.EPS_UNIT, "total")[:, 2]
    ratios.append([float(pt[0] / fe_tot[0]), float(pt[1] / fe_tot[-1])])
    print(f"  {f}: end deflection PINN/FE left {ratios[-1][0]:.3f}, right {ratios[-1][1]:.3f}")
mr = np.mean(ratios, 0)
tw = {"x": xt.tolist(), "fe_dz_total": fe_tot.tolist(), "ratios_per_seed": ratios, "ratio_left_mean": float(mr[0]), "ratio_right_mean": float(mr[1])}
json.dump(tw, open("results/fe_twin.json", "w"), indent=1)
print(f"  mean over seeds: left {mr[0]:.3f}, right {mr[1]:.3f}")
