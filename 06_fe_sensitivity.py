"""
FE sensitivity of the calibration (Section 5.4) and homogenised against explicitly resolved comb supports (Section 4.4).

    python 06_fe_sensitivity.py            # all cases (several hours on a laptop CPU, one case at a time)
    python 06_fe_sensitivity.py beam_2.3   # selected cases

For every case the FE solution of the elastic ISM is computed at unit strain, eps* is calibrated on the measured
partial-cut profile of C (Eq. 9) and D, the bow after removal and the on-plate stress are predicted with it.
Each case is stored in results/sensitivity/<case>.json; results/fe_sensitivity.json collects them.
"""
import json, os, sys
import numpy as np
import ispinn as isp
import fe_reference as fe

m = isp.Material(); X8 = np.array(json.load(open("data/experiment_dinesh2025.json"))["probe_x_mm"])
E = json.load(open("data/experiment_dinesh2025.json")); mC = np.array(E["cut_dz_mm"]["C"])
os.makedirs("results/sensitivity", exist_ok=True)


def explicit_boxes(g, tooth_t=1.0, pitch=2.0):
    """Comb teeth resolved as solid walls normal to x (void between them), solid parts as in the PINN geometry."""
    part = [b for b in g.boxes if b.role == "part"]
    sup = [b for b in g.boxes if b.role == "support"]
    out = [(b.lo, b.hi, "solid") for b in part]
    for b in sup:
        x0, x1 = b.lo[0], b.hi[0]
        n = int(np.floor((x1 - x0) / pitch))
        for i in range(n):
            c = x0 + pitch * (i + 0.5)
            out.append(((c - tooth_t / 2, b.lo[1], b.lo[2]), (c + tooth_t / 2, b.hi[1], b.hi[2]), "solid"))
    return out


def case(name):
    beam_t, teeth_h, tooth_t, pitch, layer = 2.6, 6.4, 1.0, 2.0, 0.55
    dx, explicit = 1.0, False
    cutC, cutD = 60.0, 54.0
    kind, val = name.split("_", 1) if "_" in name else (name, None)
    if kind == "beam": beam_t = float(val)
    elif kind == "tooth": tooth_t = float(val[1:])
    elif kind == "pitch": pitch = float(val)
    elif kind == "layer": layer = float(val)
    elif kind == "height": teeth_h = float(val)
    elif kind in ("homog", "explicit"): teeth_h, dx, explicit = float(val), 0.5, kind == "explicit"
    elif kind == "cutC": cutC += float(val)
    elif kind == "cutD": cutD += float(val)
    elif kind == "cutz": teeth_h -= float(val)          # cut made higher by val: shorter remaining support below the beam
    elif kind != "reference": raise ValueError(name)
    H = beam_t + teeth_h
    g = isp.paper_cantilever(H=H, beam_t=beam_t, tooth_t=tooth_t, pitch=pitch)
    nl = max(1, round(H / layer)); lh = H / nl
    boxes = explicit_boxes(g, tooth_t, pitch) if explicit else fe.boxes_from_geometry(g)
    return g, boxes, lh, (dx, 1.0, lh / 2), {"C": cutC, "D": cutD}


def run(name):
    f = f"results/sensitivity/{name}.json"
    if os.path.exists(f):
        return json.load(open(f))
    g, boxes, lh, h, cuts = case(name)
    print(f"case {name}: H {g.H:.2f} mm, layer {lh:.4f} mm, mesh {h}", flush=True)
    r = fe.run_fe(boxes, layer_h=lh, E=m.E, nu=m.nu, h=h, cuts=cuts, probe_x=X8,
                  support=g.support.voigt(m.E, m.nu), log=lambda *a: None)
    uC, uD = np.asarray(r["cut_C"]), np.asarray(r["cut_D"])
    k = float(uC @ mC / (uC @ uC))                       # eps* / eps0
    bow_u = float(np.ptp(isp.detrend(np.asarray(r["removed"]), X8)))
    out = {"case": name, "H": g.H, "layer_h": lh, "mesh_mm": list(h), "dofs": int(r["dofs"]), "time_s": float(r["time_s"]),
           "unit": {"cut_C": uC.tolist(), "cut_D": uD.tolist(), "bow": bow_u, "vm_on": float(np.mean(r["asbuilt_vm_top"]))},
           "eps_star_pct": 0.1 * k,
           "pred": {"C0": k * uC[0], "D0": k * uD[0], "D_over_C_unit": float(uD[0] / uC[0]), "bow": k * bow_u,
                    "vm_on": k * float(np.mean(r["asbuilt_vm_top"])),
                    "C_rmse": float(np.sqrt(np.mean((k * uC - mC) ** 2)))}}
    json.dump(out, open(f, "w"), indent=1)
    p = out["pred"]
    print(f"   eps* {out['eps_star_pct']:.3f} %  D0 {p['D0']:.2f}  bow {p['bow']:.2f}  vm {p['vm_on']:.0f}  ({out['time_s']:.0f} s)", flush=True)
    return out


CASES = ["reference", "beam_2.3", "beam_2.9", "tooth_t0.8", "tooth_t1.2", "pitch_1.6", "pitch_2.4", "layer_0.45",
         "layer_0.75", "height_5.4", "height_7.4", "homog_6.4", "explicit_6.4", "homog_8.4", "explicit_8.4",
         "cutC_-2", "cutC_+2", "cutD_-2", "cutD_+2", "cutz_0.5"]
if __name__ == "__main__":
    todo = sys.argv[1:] or CASES
    for c in todo:
        run(c)
    allr = {c: json.load(open(f"results/sensitivity/{c}.json")) for c in CASES if os.path.exists(f"results/sensitivity/{c}.json")}
    json.dump(allr, open("results/fe_sensitivity.json", "w"), indent=1)
