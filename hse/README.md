# hse/ — pretraining pool and shared training code

The distributional model learns its representation on Materials Project band gaps before it is fitted to the measured gaps. This folder builds that pretraining pool and holds the training code shared with `cv/build_data.py` and `model/`.

| File | What it does |
|---|---|
| `build_pool.py` | Builds the pretraining pool from `external/mp/mp_hse.csv`, keeping polymorphs within 0.1 eV per atom of the hull and removing every composition of the traced, compiled and external experimental sets, and encodes it with the Roost encoder of MatterVial-MP-gap. Writes `hse_pool.csv`, `hse_pool_counts.json` and `hse_pool_features.pt`, read by the fine-tuning preparation (`cv/repeated/prep_ft.py`) and by `cv/build_final_ledger.py` |
| `hse_common.py` | Paths and helpers shared by `cv/build_data.py`, `cv/cvlib.py` and the scripts here |
| `runtime/` | Model, training and data code of the earlier GapScope stages, unchanged; `model/gsft7.py` reads `runtime/modules/symmetry_table.json` |
| `pretrain.py`, `arm.py`, `evaluate.py` | Pretraining and scoring of the earlier comparison arms (H, H_MEAN, H_AUX, FID, ARCH). No number of the paper comes from them |

`arm.py` and `evaluate.py` also read the earlier model stages placed under `external/stage1`, `external/stage2` and `external/augmentation` (see `external/README.md`).
