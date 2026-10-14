# TempAlign-EEGNet — reproducibility package

Code accompanying the manuscript *"Temporal Predictive Alignment Between EEG
and Eye Movements for Gaze Prediction Using Cross-Attention"* (IEEE Access
resubmission, manuscript ID Access-2026-39609).

## Contents

| File | Purpose |
|---|---|
| `data_pipeline.py` | Loads the BALLADEER dataset, preprocesses EEG (detrend, 1–45 Hz band-pass, AF3/AF4 regression-based EOG correction), builds 2 s windows with 50-point uniformly resampled gaze contexts, applies EEG quality control (peak-to-peak 250 µV), and freezes a single window cache used by every experiment. |
| `models_v2.py` | TempAlign-EEGNet (fixed temporal mask), TempAlignGazeOnly, TempAlignEEGOnly, and TempStructNet (learned mask with straight-through Gumbel–Softmax sampling and a −10⁹ additive penalty; deterministic binarized evaluation mask). |
| `baselines_v2.py` | LSTM / Transformer / LTC / wavelet baselines with identical dual inputs (EEG + gaze) and a shared unmasked cross-attention fusion scheme. |
| `metrics.py` | Pearson r, out-of-sample predictive R² (computed from errors), per-participant metrics, participant-level bootstrap CIs, paired Wilcoxon / t tests. |
| `run_main.py` | Nested participant-level 5-fold CV runner: outer folds for evaluation, inner 20% participant split for early stopping; trains all models and the statistical baselines; saves per-run metrics and prediction files. |
| `run_ablations.py` | No-mask, frontal-channel-excluded (8 channels), learned-mask, single-band (theta/alpha/beta), EEG-scrambled, target-shuffled, and parameter-matched variants. |
| `run_lagged.py` | Properly lagged directional predictability analysis (16 Hz, 4 lags, blocked CV, circular-shift permutation), both directions. |
| `verify_mask.py` | Trains fold-0 models and verifies that the maximum attention weight on temporally prohibited positions is exactly zero for both mask types; synthetic non-identifiability control for the learned delay. |
| `run_channel_zeroout.py` | Per-channel zero-out importance across folds. |
| `make_figures.py`, `make_arch_figure.py` | Generate every manuscript figure from the results registry files. |
| `summarize_results.py`, `emit_numbers.py` | Aggregate all runs into `summary.json` and emit LaTeX number macros so that text, tables, and figures are numerically identical by construction. |

## Reproduction

```bash
pip install torch numpy scipy pandas scikit-learn statsmodels matplotlib

# 1. Build the frozen cache (requires the BALLADEER ADHD dataset at DATA/...)
python data_pipeline.py

# 2. Main comparison (nested participant-level 5-fold CV)
python run_main.py --models TempAlignEEGNet,GazeOnly,EEGOnly,LSTM,Transformer,LTC,WaveletTransformer

# 3. Statistical baselines
python run_main.py --models StatBaselines

# 4. Ablations and controls
python run_ablations.py --variants NoMask,FrontalExcluded,TempStructNet,PermutedTarget --seeds 42,43,44
python run_ablations.py --variants PermutedEEG,BandTheta,BandAlpha,BandBeta,LTC_param_matched --seeds 42

# 5. Mask verification, lagged analysis, channel importance
python verify_mask.py
python run_lagged.py
python run_channel_zeroout.py

# 6. Aggregate + figures
python summarize_results.py
python make_figures.py
```

The pipeline is deterministic given the dataset: preprocessing uses fixed-order
operations, window construction and fold assignment use fixed seeds
(outer split seed 42; training seeds 42/43/44), and all models in a given
comparison share folds, normalization statistics, and training protocol.
GPU kernels may introduce minor nondeterminism; exact GPU nondeterminism can
be disabled with `torch.use_deterministic_algorithms(True)` if required.

## Data

BALLADEER: A multimodal dataset for neurophysiological and AI applications.
Trujillo et al., *Scientific Data* 13, 436 (2026), CC BY 4.0.
