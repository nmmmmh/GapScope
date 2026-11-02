# GapScope public code

This repository contains the training, inference and density-scoring implementation for **GapScope**. The primary model retains the internal key `FROZEN` for compatibility with archived checkpoints: 3 seeds (`17`, `29`, `41`) × 5 composition-grouped folds. The supplied training entry also runs the paper's matched `F`, `J`, `J_SG`, and `ANCHORED` comparison arms. Later supporting diagnostics do not replace the paper's FROZEN model.

The repository contains source code and a symmetry lookup table only. It contains no experimental or auxiliary data, CIFs, encoder weights, trained checkpoints, figures, or caches. Data and trained weights must be supplied separately.

## Install

Python 3.10 is the reference runtime. Install a PyTorch build compatible with the machine first, then install this package and its remaining dependencies:

```bash
python -m pip install "torch>=2.4,<2.5"
python -m pip install .
```

The training recipe requires a CUDA device. Inference and density scoring support CPU. The runtime dependencies are declared in `pyproject.toml`: PyTorch, NumPy, pandas, and SciPy. The original GPU execution used PyTorch 2.4.0 with CUDA 12.1.

## External data interface

The training code reads a dataset root with this layout:

```text
DATA_ROOT/
└── data/
    ├── reports/
    │   ├── features.pt
    │   ├── metadata.json
    │   └── records.jsonl
    └── composition/
        ├── features.pt
        └── metadata.json
```

`features.pt` files are PyTorch dictionaries of row-aligned tensors. Required feature keys and shapes are:

| Key | Shape | Meaning |
|---|---:|---|
| `composition` | `[N, 183]` | Raw frozen composition-encoder features |
| `sg_features` | `[N, 237]` | Raw space-group features; the first 230 columns encode group identity |
| `structure` | `[N, 160]` | Raw frozen structure-encoder features |
| `sg_mask` | `[N, 1]` | Binary space-group availability |
| `structure_mask` | `[N, 1]` | Binary structure availability |
| `targets` | `[N]` | Composition-level targets in eV; required only in `composition/features.pt` |

The masks must be binary, and features and targets must be finite. Report and composition metadata are JSON lists aligned to their corresponding feature rows and must contain `composition_key`; report metadata also carries `measurement_id` and `formula`. `reports/records.jsonl` has one JSON object per report in the same order and supplies `measurement_id`, `composition_key`, `formula`, positive `gap_ev` in eV, and the source/specimen grouping fields used for hierarchical weights. The current loader is bound to the paper cohort of 205 reports and 56 compositions; changing cohorts or split rules requires a separately reviewed protocol.

The loader transforms raw `sg_features` (237 columns) into hybrid symmetry features (290 columns): 230 group-identity features plus 60 table-derived symmetry features. It then constructs the model input as `183 + 290 + 160 + 2 = 635` columns. The four training views are formula only, formula + space group, formula + structure, and all three modalities. Do not pass hybrid 290-column SG features to `training_data.build_fold`; it expects the raw 237-column representation and performs this transformation itself.

The code does not encode formulas or CIFs into the frozen 183- and 160-dimensional feature vectors. Those encoders and their weights are external prerequisites. For inference, supply the vectors in a JSON file; for training, place the row-aligned tensors and metadata in the layout above. The loader uses the supplied 237-column raw SG features as-is and does not apply candidate-SG completion overlays; supply only the feature set verified for the formal FROZEN model.

## Train

Run the fixed five-fold, three-seed recipe and write outputs outside the data root:

```bash
gapscope-train --root DATA_ROOT --work-dir RUN_DIR
```

The run directory receives one checkpoint bundle per seed and fold. It also records the data and source-code hashes. Training writes the main FROZEN state alongside its matched control states; the FROZEN state is the paper's primary model.

## Predict

Inference requires the 15 external checkpoint bundles produced by the recipe, or a compatible separately supplied release. Each must be at `CHECKPOINT_DIR/seed_<seed>_fold_<fold>/models.pt` and use schema `current_training_v1`.

The JSON must provide `composition` as an array of 183 finite numbers. It may also provide `space_group` as an integer from 1 to 230 and `structure` as an array of 160 finite numbers. `space_group` and `structure` are optional. The composition vector is required even when the optional modalities are absent. The CLI supports `formula` (composition-only view) and `multimodal` (uses whichever optional modalities are present); both use the FROZEN checkpoints.

```bash
gapscope-predict --features INPUT.json --checkpoint-dir CHECKPOINT_DIR \
  --model multimodal --output PREDICTION.json
```

The output includes the predictive density mixture, median, and central quantiles in eV. The mixture equally weights the 15 seed-fold predictions; the two lognormal components within each fold are retained. Density components are statistical mixture terms, not phase labels.

## Score

The scoring function is also available as `density_score.score(gap_ev, (weights, log_mu, log_sigma))`. The CLI accepts one positive observed gap and its mixture parameters:

```bash
gapscope-score --input SCORE_INPUT.json
```

Input JSON fields are `gap_ev`, `weights`, `log_mu`, and `log_sigma`; the three arrays contain one entry per mixture component. Reported metrics include log density NLL, CRPS, median absolute error, interval coverage, and interval width.

## Entry points

- `gapscope-train` — formal five-fold training recipe and matched comparison arms.
- `gapscope-predict` — FROZEN inference from precomputed encoder features and external checkpoints.
- `gapscope-score` — scoring for a positive-gap observation and lognormal-mixture prediction.
- Python APIs: `training_data.build_fold`, `inference.predict_features`, and `density_score.score`.

## License

See [`LICENSE_STATUS.md`](LICENSE_STATUS.md). The authors have not selected an open-source license for this package.
