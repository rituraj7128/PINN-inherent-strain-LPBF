"""
Step 2 — calibrate eps* on cantilever C and compute every prediction of the paper, for each trained seed.

    python 02_evaluate.py                 # uses models/cantilever_seed*.pt and models/twin_seed*.pt
    python 02_evaluate.py --models models_test --out results_test

Output: <out>/seed<N>.json (fields along the measurement lines) and <out>/summary.json + summary table
(mean ± standard deviation over seeds; errors against experiment and Simufact Additive).
"""
import argparse, glob, json, os, re
import numpy as np
import ispinn as isp

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="models")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

E = json.load(open("data/experiment_dinesh2025.json"))
S = json.load(open("data/simufact_twin_cantilever.json"))
X8 = np.array(E["probe_x_mm"])
gC, gT = isp.paper_cantilever(), isp.twin_cantilever()
log = lambda *m: None
def top(xs, g):                                       # 0.14 mm below the top surface (XRD depth region)
    xs = np.asarray(xs, float); return np.c_[xs, 0 * xs, 0 * xs + g.z0 + g.H - 0.14]

seeds = sorted(int(re.findall(r"seed(\d+)", f)[0]) for f in glob.glob(f"{a.models}/cantilever_seed*.pt"))
per = {}
for sd in seeds:
    c = isp.ISPINN.load(f"{a.models}/cantilever_seed{sd}.pt", gC, log=log)
    t = isp.ISPINN.load(f"{a.models}/twin_seed{sd}.pt", gT, log=log)
    P = isp.top_line(gC, X8)
    uC = c.displacement("cut_C", P, isp.EPS_UNIT, "release")[:, 2]
    eps = isp.calibrate_eps(uC, E["cut_dz_mm"]["C"])                       # calibration on C only (Eq. 9)
    r = {"seed": sd, "eps": eps}
    for k in "CD":
        r[f"cut_{k}"] = c.displacement(f"cut_{k}", P, eps, "release")[:, 2].tolist()
        r[f"vm_cut_{k}"] = isp.von_mises(c.stress(f"cut_{k}", top(X8, gC), eps)).tolist()
    S_on, S_cC = c.stress("onplate", top(X8, gC), eps), c.stress("cut_C", top(X8, gC), eps)
    r["vm_on"] = isp.von_mises(S_on).tolist()
    r["sxx_on"], r["syy_on"] = S_on[:, 0, 0].tolist(), S_on[:, 1, 1].tolist()       # x: beam axis, y: across
    r["sxx_cut_C"], r["syy_cut_C"] = S_cC[:, 0, 0].tolist(), S_cC[:, 1, 1].tolist()
    r["removed_detr"] = isp.detrend(c.displacement("removed", P, eps)[:, 2], X8).tolist()
    xt = np.linspace(0, 72, 73)
    r["twin_x"] = xt.tolist()
    r["twin_dz_cut"] = t.displacement("cut", isp.top_line(gT, xt), eps, "total")[:, 2].tolist()
    r["twin_dz_at_sim"] = t.displacement("cut", isp.top_line(gT, S["dz_x"]), eps, "total")[:, 2].tolist()
    r["twin_vm_on"] = isp.von_mises(t.stress("onplate", top(S["vm_on_x"], gT), eps)).tolist()
    r["twin_vm_cut"] = isp.von_mises(t.stress("cut", top(S["vm_cut_x"], gT), eps)).tolist()
    json.dump(r, open(f"{a.out}/seed{sd}.json", "w"), indent=1)
    per[sd] = r
    print(f"seed {sd}: eps* = {100 * eps:.3f} %", flush=True)

# ---------------- summary: the numbers quoted in the paper ----------------
pc = lambda p, e: 100 * (p - e) / e
sz = np.array(S["dz"])
def metrics(r):
    dz = np.array(r["twin_dz_at_sim"]); bow = float(np.ptp(r["removed_detr"]))
    return {"eps_pct": 100 * r["eps"],
            "C_pt0": r["cut_C"][0], "D_pt0": r["cut_D"][0],
            "C_rmse": float(np.sqrt(np.mean((np.array(r["cut_C"]) - E["cut_dz_mm"]["C"]) ** 2))),
            "D_rmse": float(np.sqrt(np.mean((np.array(r["cut_D"]) - E["cut_dz_mm"]["D"]) ** 2))),
            "bow": bow, "vm_on": float(np.mean(r["vm_on"])), "vm_cut_C": float(np.mean(r["vm_cut_C"])),
            "twin_L": dz[0], "twin_R": dz[-1], "twin_rmse": float(np.sqrt(np.mean((dz - sz) ** 2))),
            "twin_vm_on": float(np.mean(r["twin_vm_on"])), "twin_vm_cut": float(np.mean(r["twin_vm_cut"]))}
M = [metrics(per[s]) for s in seeds]
mean = {k: float(np.mean([m[k] for m in M])) for k in M[0]}
sd_ = {k: float(np.std([m[k] for m in M], ddof=1)) if len(M) > 1 else 0.0 for k in M[0]}
rows = [("eps* (%)", None, "eps_pct"),
        ("C free-end deflection (mm) [calibration]", 3.29, "C_pt0"),
        ("D free-end deflection (mm)", 2.34, "D_pt0"),
        ("Bow after removal vs CT C (mm)", 1.12, "bow"),
        ("Bow after removal vs CT D (mm)", 0.93, "bow"),
        ("On-plate stress, mean vs XRD C (MPa)", 680, "vm_on"),
        ("On-plate stress, mean vs XRD D (MPa)", 490, "vm_on"),
        ("After-cut stress C, mean vs XRD (MPa)", 446, "vm_cut_C"),
        ("Twin left end Z vs Simufact (mm)", float(sz[0]), "twin_L"),
        ("Twin right end Z vs Simufact (mm)", float(sz[-1]), "twin_R"),
        ("Twin on-plate stress vs Simufact (MPa)", float(np.mean(S["vm_on"])), "twin_vm_on"),
        ("Twin after-cut stress vs Simufact (MPa)", float(np.mean(S["vm_cut"])), "twin_vm_cut")]
table = []
print(f"\n{'quantity':44s} {'reference':>10s} {'PINN mean ± sd':>18s} {'error':>8s}")
for q, ref, k in rows:
    err = "" if ref is None else f"{pc(mean[k], ref):+.1f} %"
    print(f"{q:44s} {('' if ref is None else f'{ref:.2f}'):>10s} {mean[k]:10.3f} ± {sd_[k]:<6.3f} {err:>8s}")
    table.append({"quantity": q, "reference": ref, "mean": mean[k], "sd": sd_[k], "error_pct": None if ref is None else pc(mean[k], ref)})
print(f"{'Twin Z profile RMSE vs Simufact':44s} {'':>10s} {mean['twin_rmse']:10.3f} mm = {100 * mean['twin_rmse'] / sz.max():.1f} % of peak")
print(f"{'C / D profile RMSE vs experiment (mm)':44s} {'':>10s} {mean['C_rmse']:.3f} / {mean['D_rmse']:.3f}")
json.dump({"seeds": seeds, "per_seed": M, "mean": mean, "sd": sd_, "table": table,
           "twin_rmse_pct_of_peak": 100 * mean["twin_rmse"] / float(sz.max())}, open(f"{a.out}/summary.json", "w"), indent=1)
print(f"\nsaved {a.out}/summary.json")
