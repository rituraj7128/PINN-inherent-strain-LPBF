"""Top-surface stress components after cut D (for the XRD value of D after cutting), three seeds, at the calibrated eps*."""
import json, glob, numpy as np, ispinn as isp
g = isp.paper_cantilever(); X8 = np.array(json.load(open("data/experiment_dinesh2025.json"))["probe_x_mm"])
top = np.c_[X8, 0 * X8, 0 * X8 + g.H - 0.14]; out = []
for f in sorted(glob.glob("models/cantilever_seed*.pt")):
    m = isp.ISPINN.load(f, g, device="cpu", log=lambda *a: None); eps = json.load(open(f"results/seed{f[-4]}.json"))["eps"]
    S = m.stress("cut_D", top, eps); out.append({"sxx": float(S[:, 0, 0].mean()), "syy": float(S[:, 1, 1].mean()), "vm": float(isp.von_mises(S).mean())})
res = {k: float(np.mean([o[k] for o in out])) for k in out[0]}; res["seeds"] = out
print(res); json.dump(res, open("results/stress_cutD.json", "w"), indent=1)
