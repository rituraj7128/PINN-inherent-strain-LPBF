"""Revision runs 3 (GPU, about 6 h): seeds 1 and 2 for the remaining ablation variants, so that every row of the
ablation table has three seeds.
  release-only variants on the A0 build of the same seed:  A0+release_features, A0+tip_refinement
  new builds:  A0+stratified, A0+layer_enrichment, full-stratified, full-layer_enrichment
Outputs: models/ablation/<variant>_s<seed>.pt, times in results/ablation_v2_times.json, log results/training_logs/revision3.log"""
import json, os, time
import torch, ispinn as isp
OUT = "models/ablation"; os.makedirs(OUT, exist_ok=True)
TIMES = "results/ablation_v2_times.json"
g = isp.paper_cantilever(); m = isp.Material()
REL = {k: v for k, v in isp.SCEN_CANTILEVER.items() if v["type"] != "onplate"}
BASE = dict(enrich=False, rfeat=False, stratified=False, tip_n=0, lbfgs_release=0)
logf = open("results/training_logs/revision3.log", "a")
log = lambda *a: (print(*a, flush=True), print(*a, file=logf, flush=True))
def releases(s, name, tb, seed):
    times = json.load(open(TIMES)) if os.path.exists(TIMES) else {}
    s.release_nets, s._cloud = {}, None; s.specs = {"onplate": {"type": "onplate"}}
    torch.manual_seed(seed); t1 = time.time()
    for n, spec in REL.items(): s.train_release(n, spec)
    tr = time.time() - t1; s.save(name)
    times[os.path.basename(name)[:-3]] = {"build_s": tb, "release_s": tr, "total_min": (tb + tr) / 60}
    json.dump(times, open(TIMES, "w"), indent=1); log(f"=== {name}: build {tb/60:.1f} min, releases {tr/60:.1f} min")
for seed in (1, 2):
    # release-only features on the A0 build of this seed (build time not repeated)
    for n, ch in (("A0+release_features", {"rfeat": True}), ("A0+tip_refinement", {"tip_n": 8000})):
        f = f"{OUT}/{n}_s{seed}.pt"
        if os.path.exists(f): continue
        s = isp.ISPINN.load(f"{OUT}/A0_s{seed}.pt", g, log=log)
        for k, v in {**BASE, **ch}.items(): setattr(s.s, k, v)
        releases(s, f, float("nan"), seed)
for seed in (1, 2):
    for n, base, ch in (("A0+stratified", BASE, {"stratified": True}), ("A0+layer_enrichment", BASE, {"enrich": True}),
                        ("full-stratified", {}, {"stratified": False}), ("full-layer_enrichment", {}, {"enrich": False})):
        f = f"{OUT}/{n}_s{seed}.pt"
        if os.path.exists(f): continue
        st = isp.SolverSettings.preset("paper"); st.seed = seed
        for k, v in {**base, **ch}.items(): setattr(st, k, v)
        s = isp.ISPINN(g, m, st, log=log); torch.manual_seed(seed)
        t0 = time.time(); s.train_build(); releases(s, f, time.time() - t0, seed)
log("revision3 finished")
