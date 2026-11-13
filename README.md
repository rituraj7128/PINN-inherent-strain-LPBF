# Energy-based PINN for the calibrated layer-wise inherent strain method (LPBF)

Code, trained models and data for the manuscript

> R. Raj, *Verification and limits of a physics-informed neural network for the calibrated layer-wise inherent strain method in laser powder bed fusion* (submitted).

The PINN solves the linear-elastic, layer-wise inherent strain model (ISM) of an LPBF part, including homogenised comb supports, partial wire-EDM cuts and removal from the build plate, by minimising the elastic energy. Its training uses no simulation or measurement data. An independent finite element (FE) solver of the same model (`fe_reference.py`) is used for verification. One parameter, the inherent strain ε\*, is calibrated in closed form on a measured partial-cut profile.

## Contents

| Path | Content |
|---|---|
| `ispinn.py` | PINN: geometry, homogenised supports, networks, energy functionals, training, evaluation |
| `fe_reference.py` | Independent FE solver (scikit-fem + PyAMG) of the same model |
| `01_train.py` | Train one geometry and one seed (GPU) |
| `02_evaluate.py` | Calibrate ε\* on cantilever C and compute all predictions (`results/summary.json`, `seed*.json`) |
| `03_fe_verification.py` | PINN against FE, FE mesh study (`--refined`), twin-cantilever FE |
| `04_figures.py` | All figures of the paper |
| `05_ablation_eval.py`, `08_seed_penalty_eval.py` | Ablation table (seed 0 and three seeds), rigid-body penalty check |
| `06_fe_sensitivity.py` | FE sensitivity of the calibration (geometry, layer lumping, cut lengths, cut height) and explicit against homogenised supports |
| `09`–`14_*.py` | Earlier twin bond, stresses after cut D, beam stress profile, field errors, third mesh level, alternative calibrations |
| `run_*.py`, `START_*.bat` | GPU training runs used for the paper (ablation, seeds, penalty check) |
| `data/` | Experimental data digitised from Dinesh & Sahu (2025), with sources; Simufact Additive results for the twin cantilever |
| `models/` | Trained networks (`cantilever_seed{0,1,2}.pt`, `twin_seed{0,1,2}.pt`, `ablation/`, `penalty/`) |
| `results/` | All numerical results used in the paper (JSON) and training logs |
| `figures/` | Figures of the paper |

## Reproduce the results

```bash
pip install -r requirements.txt        # Python 3.10+, PyTorch >= 2.1 (CUDA build for training)
python 02_evaluate.py                  # calibration and predictions from the trained models (CPU, minutes)
python 03_fe_verification.py --refined # FE reference and mesh study (CPU, about 1 h, 8 GB RAM)
python 05_ablation_eval.py && python 08_seed_penalty_eval.py
python 06_fe_sensitivity.py            # FE sensitivity and support study (CPU, several hours)
python 12_field_errors.py && python 13_mesh_third.py && python 14_reverse_calibration.py
python 04_figures.py                   # figures
```

Training from scratch (GPU, about 30 min per geometry and seed on a 6 GB RTX A2000):

```bash
python 01_train.py --geometry cantilever --seed 0
```

The solver settings are listed in Appendix B of the paper and in `SolverSettings` in `ispinn.py`.

## Data sources

The cantilever geometry, process data, partial-cut profiles, CT values and XRD stresses are taken from
S. Dinesh, J. Sahu, *Thermomechanical simulation and experimental validation of distortion and residual stress for LPBF additive manufactured Inconel 718*, Mater. Today Commun. 42 (2025) 111512, https://doi.org/10.1016/j.mtcomm.2025.111512.
The profiles were digitised from the figures of that paper; the beam thickness, tooth dimensions and cut lengths were measured on its photographs (see `data/experiment_dinesh2025.json` and Table 2 of the manuscript).

## Note on the use of AI tools

The code, computations, figures and manuscript text were developed with the assistance of Claude (Anthropic), a large language model, under the direction of the author, as stated in the manuscript.

## Licence

MIT (see `LICENSE`).
