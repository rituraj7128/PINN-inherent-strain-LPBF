"""
Step 1 — train the PINN for one geometry and one random seed (unit inherent strain, no data).

    python 01_train.py --geometry cantilever --seed 0            # paper settings, GPU (~50 min)
    python 01_train.py --geometry twin --seed 0
    python 01_train.py --geometry cantilever --seed 0 --preset smoke --out models_test   # seconds, code check

In Jupyter:  %run 01_train.py --geometry cantilever --seed 0
Output: <out>/<geometry>_seed<seed>.pt  (all build and release networks + settings)
"""
import argparse, os, sys, time
import torch
import ispinn as isp

ap = argparse.ArgumentParser()
ap.add_argument("--geometry", choices=["cantilever", "twin"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--preset", choices=["paper", "cpu", "smoke"], default="paper")
ap.add_argument("--out", default="models")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", flush=True)
if dev == "cpu" and a.preset == "paper":
    print("WARNING: paper settings on a CPU take many hours; use a GPU or --preset cpu", flush=True)

geom = isp.paper_cantilever() if a.geometry == "cantilever" else isp.twin_cantilever()
scen = isp.SCEN_CANTILEVER if a.geometry == "cantilever" else isp.SCEN_TWIN
st = isp.SolverSettings.preset(a.preset); st.seed = a.seed
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"{a.geometry}_seed{a.seed}.pt")
s = isp.ISPINN(geom, isp.Material(), st, log=lambda *m: print(*m, flush=True)).train(scen)
s.save(path)
print("saved", path)
