# SHE-ABC

Code for the paper *SHE-ABC: an ensemble artificial bee colony with fast early progress, evaluated on benchmark
suites and segmentation hyperparameter tuning* (Z. Yıldırım).

SHE-ABC (success-history ensemble ABC) is an artificial bee colony in which every bee draws its move from an adaptive
ensemble of an ABC move and two differential-evolution moves, with success-history parameter adaptation and a
dimension-dependent colony. In the code it carries its development name `eshaguabc_v6`.

## Contents

| Path | What it is |
|---|---|
| `src/guabc/eshaguabc_v6.py` | SHE-ABC |
| `src/guabc/eshaguabc.py`, `eshaguabc_v2.py` … `eshaguabc_v5.py` | development versions v1–v5 (ablation on CEC 2014) |
| `src/guabc/eshaguabc_v6_ablation.py` | leave-one-out variants (`eshaguabc_v6_no_pool`, `_no_eigen`, `_no_scout`, `_no_credit`, `_no_restart`) |
| `src/guabc/optimizers.py` | baselines ABC, GABC, DE (DE/rand/1/bin) and L-SHADE, and the registry of all optimizer names |
| `src/guabc/cec2022.py` | adapter to the official CEC 2022 Python code |
| `src/guabc/cec2017.py` | NumPy port of the official CEC 2017 C code |
| `src/guabc/cec2014.py` | interface to the official CEC 2014 C code (as a DLL) |
| `src/guabc/evaluation*.py` | budgeted evaluators that record the best error at the official checkpoints |
| `scripts/` | benchmark runners, the CEC complexity procedure, the CEC 2017 port check |
| `tools/port_check/` | comparison of the CEC 2017 port with the compiled official C code |
| `configs/` | the configurations of all experiments in the paper |
| `hpo/` | hyperparameter optimization on ISPRS Potsdam (U-Net fine-tuning and post-processing) |

## Benchmark experiments

Requirements: Python 3.12, `pip install -r requirements.txt`, and the official benchmark code in `external/`
(see `external/README.md`).

```
python scripts/run_cec2022.py --config configs/cec2022_she_abc.json --workers 8
python scripts/run_cec2022.py --config configs/cec2022_competitors.json --workers 8
python scripts/run_cec2017.py --config configs/cec2017_she_abc.json --workers 8
python scripts/run_cec2017.py --config configs/cec2017_competitors.json --workers 8
python scripts/run_cec2017.py --config configs/cec2017_leave_one_out.json --workers 8
python scripts/run_cec2014.py --config configs/cec2014_versions.json --workers 8
python scripts/complexity.py
```

Each run is written to `results/<experiment>/D<d>/F<f>/<algorithm>/run_<r>.json` with its seed, the best-so-far
error at every official checkpoint and a fingerprint of the code, configuration and benchmark files. Runs use the
official seed rules and 30 runs per function (3 on CEC 2014); finished runs are skipped when a runner is restarted.

## Hyperparameter optimization (ISPRS Potsdam)

Requirements: an NVIDIA GPU, PyTorch with CUDA (for example
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`), `pip install -r hpo/requirements.txt`,
and the 2400 Potsdam patches in `data/patches` (see `data/README.md`).

```
python -m hpo.run_hpo --config configs/hpo_finetune.json --dry-run    # pipeline test without GPU
python -m hpo.run_hpo --config configs/hpo_postproc.json
python -m hpo.run_hpo --config configs/hpo_finetune.json
python -m hpo.final_test --config configs/hpo_postproc.json
python -m hpo.final_test --config configs/hpo_finetune.json
python -m hpo.analyze --results results/finetune
```

The split assigns whole source tiles to the subsets (four training tiles, one validation tile, one test tile) and is
written once to `data/split_tile.json`. Searches see only the training and validation tiles; `hpo.final_test`
evaluates the test tile once, after all searches, and retrains each selected fine-tuning configuration with its search
seed and with the training seeds in `test_retrain_seeds`. The optimizer settings are in `configs/algorithms.json`.
