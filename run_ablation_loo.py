"""Leave-one-out ablation: the full solver with ONE feature removed (complements run_ablation.py, which adds one
feature to the standard PINN). Release-only features reuse the build stage of models/cantilever_seed0.pt.
    python run_ablation_loo.py   (GPU, about 2 h) -> models/ablation/full-*.pt, results/ablation_v2_times.json"""
import json, os, time
import torch, ispinn as isp
OUT = "models/ablation"; os.makedirs(OUT, exist_ok=True); TIMES = "results/ablation_v2_times.json"
g = isp.paper_cantilever(); m = isp.Material()
REL = {k: v for k, v in isp.SCEN_CANTILEVER.items() if v["type"] != "onplate"}
times = json.load(open(TIMES)) if os.path.exists(TIMES) else {}
logf = open("results/training_logs/ablation_v2_loo.log", "a")
log = lambda *a: (print(*a, flush=True), print(*a, file=logf, flush=True))
def releases(s, name, tb):
    s.release_nets, s._cloud = {}, None; s.specs = {"onplate": {"type": "onplate"}}
    torch.manual_seed(0); t1 = time.time()
    for n, spec in REL.items(): s.train_release(n, spec)
    tr = time.time() - t1; s.save(f"{OUT}/{name}.pt")
    times[name] = {"build_s": tb, "release_s": tr, "total_min": (tb + tr) / 60}
    json.dump(times, open(TIMES, "w"), indent=1); log(f"=== {name} saved: build {tb/60:.1f} min, releases {tr/60:.1f} min")
# release-only variants on the full build
SMOKE = os.environ.get("ABL_SMOKE")
if SMOKE: OUT = SMOKE; TIMES = SMOKE + "/t.json"; os.makedirs(OUT, exist_ok=True)
def small(st):
    if SMOKE: st.n_pts = st.batch = 600; st.n_cloud = 3000; st.tip_n = min(st.tip_n, 500); st.iters_build, st.iters_release = 20, 30; st.lbfgs_release = min(st.lbfgs_release, 50)
for name, ch in (("full-lbfgs", {"lbfgs_release": 0}), ("full-release_features", {"rfeat": False}), ("full-tip_refinement", {"tip_n": 0})):
    if os.path.exists(f"{OUT}/{name}.pt"): continue
    s = isp.ISPINN.load("models/cantilever_seed0.pt", g, log=log)
    small(s.s)
    for k, v in ch.items(): setattr(s.s, k, v)
    releases(s, name, float("nan"))
# build variants
for name, ch in (("full-stratified", {"stratified": False}), ("full-layer_enrichment", {"enrich": False})):
    if os.path.exists(f"{OUT}/{name}.pt"): continue
    st = isp.SolverSettings.preset("paper"); st.seed = 0
    small(st)
    for k, v in ch.items(): setattr(st, k, v)
    s = isp.ISPINN(g, m, st, log=log); torch.manual_seed(0)
    t0 = time.time(); s.train_build(); releases(s, name, time.time() - t0)
log("leave-one-out ablation finished")
