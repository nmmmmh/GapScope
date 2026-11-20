# External inputs

`python tools/setup_paths.py` creates one folder here for each input that the scripts read and that is not
redistributed in this repository. Place the files as follows before running the scripts.

| Folder | Files | Source |
|---|---|---|
| `snumat/` | `snumat.json` (10,481 records with `SNUMAT_id`) and `hse_data.csv` (the SNUMAT table export, 13,719 rows) | SNUMAT hybrid-functional band-gap database, Kim et al., Sci. Data 2020, 7, 387 |
| `mp/` | `mp_hse.csv` (Materials Project band gaps with formula, space group, energy above hull, GGA gap and the corrected gap `target`) | Materials Project, Jain et al., APL Mater. 2013, 1, 011002; hybrid-functional correction of Chen et al., Phys. Rev. B 2024, 109 |
| `matbench/` | `expt_gap.csv` (columns `formula`, `target`; 4,604 rows) | Matbench `matbench_expt_gap`, Dunn et al., npj Comput. Mater. 2020, 6, 138 |
| `roost_release/` | the Roost release code, with `sources/extracted_members/data/datasets/expt-non-metals.csv` and the Matscholar embedding | Goodall and Lee, Nat. Commun. 2020, 11, 6280 |
| `traced_data/`, `traced_data_v1/` | the authors' extraction records (`data/confirmation/`, `data/structures/`, `source/data/reports/metadata.json`), and in `traced_data/` the MatterVial-MP-gap Roost encoder (`models/official/MatterVial-MP-gap/`) with its dependencies (`runtime/roost_deps/`) | the authors' project archive. Read by the six scripts that assemble Supplementary Data 1 (`analysis/build_sd1.py`, `analysis/benchmark/b2c_choice.py`, `analysis/benchmark/b4_budget.py`, `analysis/struct/benchsplit.py`, `analysis/struct/benchsplit_run.py`, `analysis/struct/case_check.py`), by the data and fold builders (`cv/build_data.py`, `cv/repeated/make_folds.py`) and by the pretraining pool (`hse/build_pool.py`); the analysis scripts start from `data/` |
| `stage1/`, `stage2/`, `augmentation/` | earlier model stages of the project (`challenge_identities.csv`, `challenge/`, `training/`, `inputs/`) | the authors' project archive. `hse/build_pool.py` reads `stage1/challenge_identities.csv` to exclude the experimental compositions from the pool; the rest is read only by the earlier comparison arms (`hse/arm.py`, `hse/evaluate.py`, `cv/cvlib.py`) |

The original location of every input is listed in `tools/paths_original.json`.
