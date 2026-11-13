"""Seed repeatability of the key ablation variants (seeds 0, 1, 2) and the rigid-body penalty check.
Writes results/ablation_seeds.json (mean, sd and per-seed PINN/FE ratios) and results/penalty_check.json."""
import json, glob, os, numpy as np, ispinn as isp
g = isp.paper_cantilever(); X8 = 2 + 9.7 * np.arange(8); q = lambda *a: None
FE = json.load(open(sorted(glob.glob("results/fe_cantilever_1.0x1.0x0.28*.json"))[0]))
P = isp.top_line(g, X8); top = np.c_[X8, 0 * X8, 0 * X8 + g.H - 0.14]
mC = np.array(json.load(open("data/experiment_dinesh2025.json"))["cut_dz_mm"]["C"])


def ratios(f):
    s = isp.ISPINN.load(f, g, device="cpu", log=q)
    c = s.displacement("cut_C", P, isp.EPS_UNIT, "release")[:, 2]; d = s.displacement("cut_D", P, isp.EPS_UNIT, "release")[:, 2]
    bow = np.ptp(isp.detrend(s.displacement("removed", P, isp.EPS_UNIT)[:, 2], X8))
    vm = np.mean(isp.von_mises(s.stress("onplate", top, isp.EPS_UNIT)))
    k = float(c @ mC / (c @ c))
    return {"cut_C": c[0] / FE["cut_C"][0], "cut_D": d[0] / FE["cut_D"][0], "bow": bow / np.ptp(FE["removed_detr"]),
            "vm_on": vm / np.mean(FE["vm_top_onplate"]), "eps_cal_pct": 0.1 * k, "D_cal": k * d[0]}


def path(name, seed):
    if name == "full":
        return f"models/cantilever_seed{seed}.pt"
    return f"models/ablation/{name}.pt" if seed == 0 else f"models/ablation/{name}_s{seed}.pt"


out = {}
for name in ("A0", "A0+lbfgs", "full", "full-lbfgs", "full-release_features", "full-tip_refinement"):
    per = {sd: ratios(path(name, sd)) for sd in (0, 1, 2) if os.path.exists(path(name, sd))}
    if not per:
        continue
    keys = list(next(iter(per.values())))
    out[name] = {"n": len(per), "seeds": per,
                 "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in keys},
                 "sd": {k: float(np.std([v[k] for v in per.values()], ddof=1)) if len(per) > 1 else 0.0 for k in keys}}
    mu, sdv = out[name]["mean"], out[name]["sd"]
    print(f"{name:22s} n={len(per)}  C {mu['cut_C']:.3f}±{sdv['cut_C']:.3f}  D {mu['cut_D']:.3f}±{sdv['cut_D']:.3f}  "
          f"bow {mu['bow']:.3f}±{sdv['bow']:.3f}  vm {mu['vm_on']:.3f}±{sdv['vm_on']:.3f}")
json.dump(out, open("results/ablation_seeds.json", "w"), indent=1, default=float)

# rigid-body penalty: bow of the removal state for w = 1, 10 (paper), 100, seed 0
pen = {}
for w, f in ((1, "models/penalty/removed_rb1.pt"), (10, "models/cantilever_seed0.pt"), (100, "models/penalty/removed_rb100.pt")):
    if not os.path.exists(f):
        continue
    s = isp.ISPINN.load(f, g, device="cpu", log=q)
    u = s.displacement("removed", P, isp.EPS_UNIT)
    pen[str(w)] = {"bow_unit": float(np.ptp(isp.detrend(u[:, 2], X8))), "mean_uz_unit": float(np.mean(u[:, 2]))}
if "10" in pen:
    for w in pen:
        pen[w]["bow_dev_pct"] = 100 * (pen[w]["bow_unit"] / pen["10"]["bow_unit"] - 1)
print("penalty:", pen)
json.dump(pen, open("results/penalty_check.json", "w"), indent=1)
