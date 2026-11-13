"""Retrain the corrected cantilever (9 mm part on the 2 mm solid extrusion, 16 layers of 0.5625 mm), seeds 0-2, one after another.
Run from this folder:  python run_cantilever_v2.py   -> models/cantilever_seed{0,1,2}.pt, log in results/training_logs/"""
import subprocess, sys, os, time
os.makedirs("results/training_logs", exist_ok=True)
for sd in (0, 1, 2):
    t0 = time.time()
    with open(f"results/training_logs/cantilever_v2_seed{sd}.log", "w") as f:
        r = subprocess.run([sys.executable, "-u", "01_train.py", "--geometry", "cantilever", "--seed", str(sd)], stdout=f, stderr=subprocess.STDOUT)
    print(f"seed {sd} finished, exit {r.returncode}, {(time.time() - t0) / 60:.0f} min", flush=True)
