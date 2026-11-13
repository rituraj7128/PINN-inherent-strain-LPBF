"""Revision runs (GPU, about 3 h):
  1. Seeds 1 and 2 for the key ablation variants: A0, A0+lbfgs (new A0 build per seed), and
     full-lbfgs, full-release_features, full-tip_refinement (release stage retrained on cantilever_seed{1,2}.pt).
  2. Rigid-body penalty check: removal of cantilever_seed0.pt retrained with rb_weight 1 and 100 (paper: 10).
Outputs: models/ablation/<variant>_s<seed>.pt, models/penalty/removed_rb<w>.pt, results/ablation_v2_times.json"""
import json, os, time
import torch, ispinn as isp
OUT = "models/ablation"; os.makedirs(OUT, exist_ok=True); os.makedirs("models/penalty", exist_ok=True)
TIMES = "results/ablation_v2_times.json"; times = json.load(open(TIMES)) if os.path.exists(TIMES) else {}
g = isp.paper_cantilever(); m = isp.Material()
REL = {k: v for k, v in isp.SCEN_CANTILEVER.items() if v["type"] != "onplate"}
BASE = dict(enrich=False, rfeat=False, stratified=False, tip_n=0, lbfgs_release=0)
logf = open("results/training_logs/revision2.log", "a")
log = lambda *a: (print(*a, flush=True), print(*a, file=logf, flush=True))
def releases(s, name, tb, seed, specs=REL):
    s.release_nets, s._cloud = {}, None; s.specs = {"onplate": {"type": "onplate"}}
    torch.manual_seed(seed); t1 = time.time()
    for n, spec in specs.items(): s.train_release(n, spec)
    tr = time.time() - t1; s.save(name)
    times[os.path.basename(name)[:-3]] = {"build_s": tb, "release_s": tr, "total_min": (tb + tr) / 60}
    json.dump(times, open(TIMES, "w"), indent=1); log(f"=== {name}: build {tb/60:.1f} min, releases {tr/60:.1f} min")
for seed in (1, 2):
    # A0 build and its two release variants
    todo = {n: ch for n, ch in (("A0", {}), ("A0+lbfgs", {"lbfgs_release": 1500})) if not os.path.exists(f"{OUT}/{n}_s{seed}.pt")}
    if todo:
        st = isp.SolverSettings.preset("paper"); st.seed = seed
        for k, v in BASE.items(): setattr(st, k, v)
        s = isp.ISPINN(g, m, st, log=log); torch.manual_seed(seed)
        t0 = time.time(); s.train_build(); tb = time.time() - t0
        for n, ch in todo.items():
            for k, v in ch.items(): setattr(s.s, k, v)
            releases(s, f"{OUT}/{n}_s{seed}.pt", tb, seed)
            for k in ch: setattr(s.s, k, BASE[k])
    # leave-one-out release variants on the full model of this seed
    for n, ch in (("full-lbfgs", {"lbfgs_release": 0}), ("full-release_features", {"rfeat": False}), ("full-tip_refinement", {"tip_n": 0})):
        f = f"{OUT}/{n}_s{seed}.pt"
        if os.path.exists(f): continue
        s = isp.ISPINN.load(f"models/cantilever_seed{seed}.pt", g, log=log)
        for k, v in ch.items(): setattr(s.s, k, v)
        releases(s, f, float("nan"), seed)
# rigid-body penalty check (removal only)
for w in (1.0, 100.0):
    f = f"models/penalty/removed_rb{w:g}.pt"
    if os.path.exists(f): continue
    s = isp.ISPINN.load("models/cantilever_seed0.pt", g, log=log); s.s.rb_weight = w
    releases(s, f, float("nan"), 0, {"removed": REL["removed"]})
log("revision2 finished")
