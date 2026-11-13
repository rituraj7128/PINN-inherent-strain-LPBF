"""Evaluate the ablation models against FE (unit strain): PINN/FE ratios for cut C, cut D, bow and on-plate stress,
plus training time. Writes results/ablation_v2.json."""
import json, glob, os, numpy as np, ispinn as isp
g = isp.paper_cantilever(); X8 = 2 + 9.7 * np.arange(8); q = lambda *a: None
FE = json.load(open(sorted(glob.glob("results/fe_cantilever_1.0x1.0x0.28*.json"))[0]))
T = json.load(open("results/ablation_v2_times.json")) if os.path.exists("results/ablation_v2_times.json") else {}
order = ["A0", "A0+stratified", "A0+lbfgs", "A0+layer_enrichment", "A0+release_features", "A0+tip_refinement", "full",
         "full-stratified", "full-lbfgs", "full-layer_enrichment", "full-release_features", "full-tip_refinement"]
mC = np.array(json.load(open("data/experiment_dinesh2025.json"))["cut_dz_mm"]["C"])
P = isp.top_line(g, X8); top = np.c_[X8, 0 * X8, 0 * X8 + g.H - 0.14]; out = {}
for name in order:
    f = "models/cantilever_seed0.pt" if name == "full" else f"models/ablation/{name}.pt"
    if not os.path.exists(f): continue
    s = isp.ISPINN.load(f, g, device="cpu", log=q)
    c = s.displacement("cut_C", P, isp.EPS_UNIT, "release")[:, 2]; d = s.displacement("cut_D", P, isp.EPS_UNIT, "release")[:, 2]
    bow = np.ptp(isp.detrend(s.displacement("removed", P, isp.EPS_UNIT)[:, 2], X8))
    vm = np.mean(isp.von_mises(s.stress("onplate", top, isp.EPS_UNIT)))
    k = float(c @ mC / (c @ c))                                  # eps*/eps0 calibrated on C with this variant
    out[name] = {"cut_C": c[0] / FE["cut_C"][0], "cut_D": d[0] / FE["cut_D"][0], "bow": bow / np.ptp(FE["removed_detr"]),
                 "eps_cal_pct": 0.1 * k, "D_cal": k * d[0], "bow_cal": k * bow, "vm_cal": k * vm,
                 "vm_on": vm / np.mean(FE["vm_top_onplate"]),
                 "train_min": T.get(name, {}).get("total_min", 30.2 if name == "full" else float("nan")),
                 "release_min": T.get(name, {}).get("release_s", float("nan")) / 60}
    print(f"{name:22s} C {out[name]['cut_C']:.3f}  D {out[name]['cut_D']:.3f}  bow {out[name]['bow']:.3f}  vm {out[name]['vm_on']:.3f}  {out[name]['train_min']:.0f} min")
json.dump(out, open("results/ablation_v2.json", "w"), indent=1, default=float)
