# Smoke verification

The package was rebuilt from a temporary staging copy as `gapscope_public_code-0.1.0-py3-none-any.whl` (56,148 bytes; SHA-256 `ae69b958c1637ac65e8efd2069c3dd5774b9e57efc99a215a3d6b418b2e872ae`). It was installed into a temporary prefix; the wheel is not part of this source-only delivery.

All checks ran in one Python 3.12.7 environment with PyTorch 2.4.1, pandas 2.2.3, and temporary-target NumPy 1.26.4 and SciPy 1.14.1. The temporary target also supplied the missing `python-dateutil`, `six`, and `tzdata` imports. The base environment was not modified.

- Imported `train`, `training_data`, `inference`, `density_score`, and `runtime`; `training_data.build_fold` is available and the recipe identifies FROZEN as primary.
- Transformed one synthetic raw SG row from 237 to 290 columns and constructed four views with shape `[4, 1, 635]`.
- Ran an untrained, randomly initialized instance of the deployed FROZEN architecture on those synthetic views. It returned finite two-component densities for all four views, with positive scales and normalized weights.
- Scored one synthetic prediction against an arbitrary positive target of 1.5 eV: NLL `1.7119470307622553`, CRPS `0.48716369531946935` eV, median `0.9549806090203465` eV. These values are smoke outputs only and do not measure model performance.
- The installed `gapscope-train`, `gapscope-predict`, and `gapscope-score` console commands each returned `--help` successfully.

No dataset, encoder feature file, trained checkpoint, or sealed target was loaded. No training or evaluation run was performed. The model forward and score use synthetic inputs only.

