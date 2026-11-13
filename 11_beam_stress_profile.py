"""As-built axial stress across the beam at x = 30 mm: rms deviation of PINN variants from the FE element means (unit strain)."""
import json, numpy as np, torch, ispinn as isp
g = isp.paper_cantilever(); FE = json.load(open("results/fe_cantilever_1.0x1.0x0.2812.json"))
z = np.array(FE["profile_z"]); sx = np.array(FE["profile_sxx"]); m = z > 6.4; out = {"fe_rms_MPa": float(np.sqrt(np.mean(sx[m] ** 2)))}
for k, f in (("A0", "models/ablation/A0.pt"), ("A0+layer_enrichment", "models/ablation/A0+layer_enrichment.pt"),
             ("full", "models/cantilever_seed0.pt"), ("full-layer_enrichment", "models/ablation/full-layer_enrichment.pt")):
    M = isp.ISPINN.load(f, g, device="cpu", log=lambda *a: None)
    _, S = M.asbuilt(torch.tensor(np.c_[0 * z[m] + 30.25, 0 * z[m] + 0.5, z[m]], dtype=torch.float32))
    out[k] = float(np.sqrt(np.mean((S[:, 0, 0].numpy() * M.MPa - sx[m]) ** 2)))
print(out); json.dump(out, open("results/beam_stress_profile.json", "w"), indent=1)
