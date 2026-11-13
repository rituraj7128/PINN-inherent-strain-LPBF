"""Alternative calibrations (three seeds, PINN; and FE reference mesh): eps* fitted on D (predict C), on C and D jointly,
compared with the paper's calibration on C.  ->  results/reverse_calibration.json"""
import json, glob, numpy as np, ispinn as isp
g = isp.paper_cantilever(); E = json.load(open("data/experiment_dinesh2025.json")); X8 = np.array(E["probe_x_mm"])
mC, mD = np.array(E["cut_dz_mm"]["C"]), np.array(E["cut_dz_mm"]["D"]); bC, bD = E["ct_bow_reported_mm"]["C"], E["ct_bow_reported_mm"]["D"]
P = isp.top_line(g, X8); out = {"seeds": {}}
def fits(uC, uD, bow):
    r = {}
    for name, k in (("on_C", uC @ mC / (uC @ uC)), ("on_D", uD @ mD / (uD @ uD)),
                    ("on_C_and_D", (uC @ mC + uD @ mD) / (uC @ uC + uD @ uD))):
        r[name] = {"eps_pct": 0.1 * k, "C0": k * uC[0], "D0": k * uD[0], "C_err_pct": 100 * (k * uC[0] / mC[0] - 1),
                   "D_err_pct": 100 * (k * uD[0] / mD[0] - 1), "rmse_C": float(np.sqrt(np.mean((k * uC - mC) ** 2))),
                   "rmse_D": float(np.sqrt(np.mean((k * uD - mD) ** 2))), "bow": k * bow,
                   "bowC_err_pct": 100 * (k * bow / bC - 1), "bowD_err_pct": 100 * (k * bow / bD - 1)}
    return r
for f in sorted(glob.glob("models/cantilever_seed*.pt")):
    s = isp.ISPINN.load(f, g, device="cpu", log=lambda *a: None); e0 = isp.EPS_UNIT
    uC = s.displacement("cut_C", P, e0, "release")[:, 2]; uD = s.displacement("cut_D", P, e0, "release")[:, 2]
    bow = float(np.ptp(isp.detrend(s.displacement("removed", P, e0)[:, 2], X8)))
    out["seeds"][f] = fits(uC, uD, bow)
FE = json.load(open(sorted(glob.glob("results/fe_cantilever_1.0x1.0x0.2812*.json"))[0]))
out["fe"] = fits(np.array(FE["cut_C"]), np.array(FE["cut_D"]), float(np.ptp(FE["removed_detr"])))
out["mean"] = {c: {k: float(np.mean([v[c][k] for v in out["seeds"].values()])) for k in next(iter(out["seeds"].values()))[c]} for c in ("on_C", "on_D", "on_C_and_D")}
for c, v in out["mean"].items(): print(c, {k: round(x, 3) for k, x in v.items()})
json.dump(out, open("results/reverse_calibration.json", "w"), indent=1)
