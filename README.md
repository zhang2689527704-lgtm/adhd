# TempAlign-EEGNet — EEG–gaze temporal predictive alignment

Code, results, and predictions accompanying the manuscript:

> **Temporal Predictive Alignment Between EEG and Eye Movements for Gaze
> Prediction Using Cross-Attention** (IEEE Access, manuscript ID
> Access-2026-39609).

The central finding is a rigorously controlled negative result: under matched
inputs, folds, and training budgets, the incremental predictive value of
14-channel consumer EEG beyond gaze history is small in a structured
grid task — a plain ridge regression on gaze history alone matches the full
cross-attention architecture — while the temporally masked cross-attention
mechanism itself is mathematically verified to assign exactly zero attention
to prohibited (future) positions.

## Repository layout

| Directory | Contents |
|---|---|
| `code/` | Full analysis pipeline (data loading, models, training, evaluation, figures). See `code/README.md` for the reproduction order. |
| `results/` | All experiment registries: per-fold/per-seed metrics (`results_*.json`), the aggregated `summary.json` with confidence intervals and paired tests, mask verification (`mask_verification.json`, zero-attention proof for both mask types), lagged directional analysis, channel importance, exploratory group analysis, and the LaTeX number macros (`numbers.tex`) that bind the manuscript text to these files. |
| `predictions/` | Held-out per-window predictions for every model/fold/seed (`.npz`, keys: `targets`, `preds`, `pids`) — sufficient to recompute every metric, confidence interval, and paired test in the paper. |
| `figures/` | Generated manuscript figures. |

## Reproducing

Requires Python 3.11+, PyTorch (CUDA optional), numpy, scipy, pandas,
scikit-learn, statsmodels, matplotlib.

```bash
pip install torch numpy scipy pandas scikit-learn statsmodels matplotlib

# 1. rebuild the frozen window cache from the public BALLADEER dataset
#    (deterministic; place the dataset at
#     DATA/BALLADEER ADHD DATASET/BALLADEER ADHD DATASET/ first)
python code/data_pipeline.py

# 2. main comparison (nested participant-level 5-fold CV, 126 participants)
python code/run_main.py --models TempAlignEEGNet,GazeOnly,EEGOnly,LSTM,Transformer,LTC,WaveletTransformer
python code/run_main.py --models StatBaselines

# 3. ablations and controls
python code/run_ablations.py --variants NoMask,FrontalExcluded,TempStructNet,PermutedTarget --seeds 42,43,44
python code/run_ablations.py --variants PermutedEEG,BandTheta,BandAlpha,BandBeta,LTC_param_matched --seeds 42

# 4. mask verification, lagged analysis, channel importance
python code/verify_mask.py
python code/run_lagged.py
python code/run_channel_zeroout.py

# 5. aggregate everything, then regenerate figures and macros
python code/summarize_results.py
python code/make_figures.py
python code/emit_numbers.py
```

The pipeline is deterministic given the dataset (fixed-order preprocessing,
fixed outer-split seed 42, training seeds 42/43/44); the window cache is
excluded from this repository because of its size but is regenerated exactly
by `code/data_pipeline.py`.

## Key results (see `results/summary.json` for exact values)

| Model | held-out r (Δcol) |
|---|---|
| TempAlign-EEGNet (full) | 0.221 |
| Gaze-only (same architecture) | 0.207 |
| EEG-only (same architecture) | 0.034 |
| Ridge regression (gaze history only) | 0.218 |

Paired participant-level tests (n = 118): full vs. gaze-only
Δr = −0.016 (p = 0.17); full vs. ridge Δr = −0.008 (p = 0.51); full vs.
EEG-only Δr = +0.203 (p ≈ 7 × 10⁻²⁰). Verified maximum attention on
temporally prohibited positions: 0.0 for both mask types.

## Data

BALLADEER: *A multimodal dataset for neurophysiological and AI applications*,
Trujillo et al., Scientific Data 13, 436 (2026), CC BY 4.0.
