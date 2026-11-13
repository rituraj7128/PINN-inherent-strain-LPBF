"""Twin cantilever with the bond interval of the first version (31-41 mm, models_v2_twin_bond31_41): end errors vs Simufact."""
import json, glob, numpy as np, ispinn as isp
S = json.load(open("data/simufact_twin_cantilever.json")); gT = isp.twin_cantilever()
eps = np.mean([json.load(open(f))["eps"] for f in sorted(glob.glob("results/seed*.json"))])
L, Rr = [], []
for f in sorted(glob.glob("models_v2_twin_bond31_41/twin_seed*.pt")):
    t = isp.ISPINN.load(f, gT, device="cpu", log=lambda *a: None)
    print(f, t.specs["cut"])
    dz = t.displacement("cut", isp.top_line(gT, S["dz_x"]), eps, "total")[:, 2]
    L.append(dz[0]); Rr.append(dz[-1])
out = {"L_err_pct": float(100 * (np.mean(L) / S["dz"][0] - 1)), "R_err_pct": float(100 * (np.mean(Rr) / S["dz"][-1] - 1)), "n": len(L)}
print(out); json.dump(out, open("results/twin_old_bond.json", "w"), indent=1)
