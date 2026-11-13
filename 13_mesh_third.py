"""Observed order of convergence along the beam. Mesh sizes must align with the geometry boxes and cut positions
(0, 3, 54, 60, 62, 72 mm), so the sequence is 1.0, 0.5, 0.25 mm along x; to fit in memory it is run with 2 mm elements
across the width. The observed order p is then used for the Richardson extrapolation of the reference-width meshes
(1.0 and 0.5 mm along x, 1 mm across).  ->  results/mesh_convergence3.json"""
import json, os, numpy as np, ispinn as isp, fe_reference as fe
g = isp.paper_cantilever(); m = isp.Material(); lh = g.H / round(g.H / 0.55); hz = lh / 2
X8 = np.array(json.load(open("data/experiment_dinesh2025.json"))["probe_x_mm"])
q = lambda r: {"c": float(np.asarray(r["cut_C"])[0]), "d": float(np.asarray(r["cut_D"])[0]),
               "bow": float(np.ptp(isp.detrend(np.asarray(r["removed"]), X8))) if "removed" in r else float(np.ptp(r["removed_detr"]))}
seq = {}
for dx in (1.0, 0.5, 0.25):
    fn = f"results/fe_mesh_x{dx}_y2.json"
    if not os.path.exists(fn):
        r = fe.run_fe(fe.boxes_from_geometry(g), layer_h=lh, E=m.E, nu=m.nu, h=(dx, 2.0, hz), cuts={"C": 60.0, "D": 54.0}, probe_x=X8,
                      support=g.support.voigt(m.E, m.nu), log=lambda *a: None)
        json.dump({"mesh": [dx, 2.0, hz], "dofs": int(r["dofs"]), **q(r), "time_s": r["time_s"]}, open(fn, "w"), indent=1)
    seq[dx] = json.load(open(fn)); print(dx, seq[dx], flush=True)
F1 = json.load(open(f"results/fe_cantilever_1.0x1.0x{hz:.4g}.json")); F2 = json.load(open(f"results/fe_cantilever_0.5x1.0x{hz:.4g}.json"))
ref = {"c": F1["cut_C"][0], "d": F1["cut_D"][0], "bow": float(np.ptp(F1["removed_detr"]))}
fin = {"c": F2["cut_C"][0], "d": F2["cut_D"][0], "bow": float(np.ptp(F2["removed_detr"]))}
out = {"sequence_y2": seq}
for k in ("c", "d", "bow"):
    f1, f2, f3 = seq[1.0][k], seq[0.5][k], seq[0.25][k]
    ratio = (f2 - f1) / (f3 - f2) if f3 != f2 else float("nan")
    p = float(np.log(ratio) / np.log(2)) if ratio > 0 else float("nan")
    fx_y2 = f3 + (f3 - f2) / (2 ** p - 1) if p == p else float("nan")
    gci = 1.25 * abs((f3 - f2) / f3) / (2 ** p - 1) if p == p else float("nan")
    fx = fin[k] + (fin[k] - ref[k]) / (2 ** p - 1) if p == p else float("nan")      # reference-width meshes, observed order
    out[k] = {"order": p, "extrapolated_y2": fx_y2, "gci_fine_y2_pct": 100 * gci, "extrapolated": fx}
print(out); json.dump(out, open("results/mesh_convergence3.json", "w"), indent=1)
