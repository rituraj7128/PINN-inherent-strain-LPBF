"""Retrain the twin cantilever with the bond interval of the Simufact run (29.0-41.85 mm), seeds 0-2.
Old twin models (bond 31-41 mm) are moved to models_v2_twin_bond31_41/."""
import subprocess, sys, os, time, shutil
os.makedirs("models_v2_twin_bond31_41", exist_ok=True); os.makedirs("results/training_logs", exist_ok=True)
for sd in (0, 1, 2):
    f = f"models/twin_seed{sd}.pt"
    if os.path.exists(f) and not os.path.exists(f"models_v2_twin_bond31_41/twin_seed{sd}.pt"):
        shutil.copy(f, f"models_v2_twin_bond31_41/twin_seed{sd}.pt")
for sd in (0, 1, 2):
    t0 = time.time()
    with open(f"results/training_logs/twin_v3_seed{sd}.log", "w") as fo:
        r = subprocess.run([sys.executable, "-u", "01_train.py", "--geometry", "twin", "--seed", str(sd)], stdout=fo, stderr=subprocess.STDOUT)
    print(f"twin seed {sd} finished, exit {r.returncode}, {(time.time() - t0) / 60:.0f} min", flush=True)
