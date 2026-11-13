"""One-factor-at-a-time ablation of the solver features (Section 4.2), corrected cantilever geometry.

Baseline A0 = standard energy PINN: uniform points, no layer-interface feature, no release features,
no cut-tip refinement, Adam only. Each variant adds ONE feature to A0. The full solver (A_full) is
models/cantilever_seed0.pt. Same seed (0), network size and iteration counts for every variant.

    python run_ablation.py          (GPU, about 2 h)  -> models/ablation/*.pt, results/ablation_v2_times.json
Then (CPU):  python 05_ablation_eval.py  -> results/ablation_v2.json
"""
import json, os, time, copy
import torch, ispinn as isp

OUT = os.environ.get("ABL_OUT", "models/ablation"); os.makedirs(OUT, exist_ok=True); os.makedirs("results/training_logs", exist_ok=True)
g = isp.paper_cantilever(); m = isp.Material()
REL = {k: v for k, v in isp.SCEN_CANTILEVER.items() if v["type"] != "onplate"}
BASE = dict(enrich=False, rfeat=False, stratified=False, tip_n=0, lbfgs_release=0)
# build variants -> release variants trained on that build
PLAN = {"A0":       (dict(),                  {"A0": {}, "A0+lbfgs": {"lbfgs_release": 1500},
                                                "A0+release_features": {"rfeat": True}, "A0+tip_refinement": {"tip_n": 8000}}),
        "strat":    (dict(stratified=True),   {"A0+stratified": {}}),
        "enrich":   (dict(enrich=True),       {"A0+layer_enrichment": {}})}
TIMES = os.environ.get("ABL_TIMES", "results/ablation_v2_times.json")
times = json.load(open(TIMES)) if os.path.exists(TIMES) else {}
logf = open("results/training_logs/ablation_v2.log", "a")
log = lambda *a: (print(*a, flush=True), print(*a, file=logf, flush=True))
for bname, (bset, rels) in PLAN.items():
    todo = [r for r in rels if not os.path.exists(f"{OUT}/{r}.pt")]
    if not todo:
        continue
    st = isp.SolverSettings.preset(os.environ.get("ABL_PRESET", "paper")); st.seed = 0
    for k, v in {**BASE, **bset}.items(): setattr(st, k, v)
    s = isp.ISPINN(g, m, st, log=log)
    torch.manual_seed(0)
    t0 = time.time(); s.train_build(); tb = time.time() - t0
    for rname in todo:
        s.release_nets, s._cloud = {}, None
        s.specs = {"onplate": {"type": "onplate"}}
        for k, v in rels[rname].items(): setattr(s.s, k, v)
        torch.manual_seed(0); t1 = time.time()
        for name, spec in REL.items():
            s.train_release(name, spec)
        tr = time.time() - t1
        s.train_time = tb + tr
        s.save(f"{OUT}/{rname}.pt")
        times[rname] = {"build_s": tb, "release_s": tr, "total_min": (tb + tr) / 60}
        json.dump(times, open(TIMES, "w"), indent=1)
        log(f"=== {rname} saved: build {tb/60:.1f} min, releases {tr/60:.1f} min")
        for k in rels[rname]: setattr(s.s, k, BASE[k])
log("ablation finished")
