# GapScope

Code and data for "Polymorphism Hides an Error Floor in Band-Gap Benchmarks" (Ni, Liu, Chen and Lin, submitted to the Journal of Chemical Theory and Computation).

The repository holds three things:
- the traced experimental band gaps and the polymorph gap spreads, as released with the paper;
- the distributional model and the grouped cross-validation;
- every analysis script behind a number, table or figure of the paper.

[BOUNDARIES.md](BOUNDARIES.md) lists where the results hold and where they do not, with the numbers and the scripts.

A few scripts from earlier versions of the analysis are kept for comparison. Their first lines start with `# SUPERSEDED` and name the script that replaced them, and no number, table or figure of the paper comes from them.

## Contents

| Folder | What it holds |
|---|---|
| `data/` | Supplementary Data 1 (237 traced reports of 85 compositions, with source, specimen and phase) Supplementary Data 2 (polymorph gap spread *D*<sub>c</sub> and window flags for 505 SNUMAT compositions) and Supplementary Data 3 (SNUMAT polymorphs, *D*<sub>c</sub> and computed floor for the 4,604 compositions of the Matbench `expt_gap` task) |
| `model/` | GapScope, the distributional model of the paper (`gsft.py`, with `gsft4.py` for composition holdout and `gsft7.py` for chemical-family holdout), the Roost wrapper (`roostlib.py`, `original_graph_adapter.py`), the Roost-cal inputs (`roost_cal_pred.py`) and the run environment (`env.sh`) |
| `cv/` | Data tables and fold assignments (`build_data.py`, `folds_comp.csv`, `folds_chem.csv`), Roost training (`run_roost_cv.py`), EasyUQ (`easyuq.py`), evaluation (`evaluate_cv.py`, `evaluate_cv_v4.py`, `run_eval_v4.py`) and paired bootstrap (`paired_bootstrap_final.py`) |
| `hse/` | The Materials Project pretraining pool (`build_pool.py`), the shared training code (`hse_common.py`, `runtime/`) and the earlier comparison arms (see its README) |
| `cv/repeated/` | The four further fold assignments and their summary over five assignments (see its README) |
| `analysis/` | Scripts behind the paper, by topic (below) |
| `tools/` | `setup_paths.py`, which sets up the paths for a new machine in one step, and `paths_original.json`, the original location of every path |
| `external/` | Where the external inputs go (see its README) |
| `legacy_frozen_model_2026-10/` | The earlier fixed-encoder model previously published here; it is not the model of this paper |

## Where each result comes from

| Paper | Scripts |
|---|---|
| Traced reports, within-study pairs, variance split (Section 2.1, S1–S2) | `analysis/build_sd1.py`, `analysis/struct/` |
| Error floor, curation, model errors on the floor (Section 2.2, S3) | `analysis/refcheck/b4_checks.py`, `b5_removed.py`, `b6_removed_boot.py`, `b7_compiled_removed.py`, `k8_overlap.py`, `c9_rank.py`, `analysis/figs_v4/fig2.py` |
| Hybrid-functional benchmark (Table S5) | `analysis/benchmark/`, `analysis/refcheck/b9_split_dc.py` |
| Coordination split, polymorph gap spread *D*<sub>c</sub>, computed floor, cheaper spreads (Section 2.3, S4) | `analysis/refcheck/e1_energyonly.py`, `e4_compfloor.py`, `k5_floor_theory.py` (bound, eq 3), `c8_resource_E.py`, `c14_energy.py`, `c15_write.py`, `analysis/compfloor/`, `analysis/descr/` |
| Matbench computed floor and Supplementary Data 3 (Section 2.3, S4) | `analysis/refcheck/e8s_sd3_snumat.py`, `e9_robust.py` |
| Benchmark of Borlido et al. (Section 2.3, S4) | `analysis/borlido/` |
| Polymorph pairs, local descriptors, Madelung test (Section 2.4, S5) | `analysis/struct/pairs_analysis.py`, `analysis/refcheck/m1_madelung.py`, `e3_pairs.py`, `e5_ionic.py`, `analysis/maps2d/` |
| GapScope (distributional model), coverage, CRPS, conditional and conformal calibration (Section 2.5, S6) | `model/`, `cv/`, `cv/repeated/`, `analysis/refcheck/k1_conformal.py`, `k2_conformal_full.py`, `k4_mondrian.py`, `k9_nosg_cov.py`, `k10_nosg_rank.py` (space group withheld) |
| Band-gap windows and the computed screen (Section 2.6, S7) | `analysis/window/`, `analysis/refcheck/c7_screen2.py`, `k3b_window.py`, `c16_rcs_sum.py` |
| Figures | `analysis/figs_v4/` (Figure 5b from `fig5b_new.py`, placed in the PowerPoint assembly of Figure 5; `fig3.py` and the TOC graphic `toc_v7_curation.py` read `descr/d2_per_composition_energy.csv`, written by `analysis/refcheck/c15_write.py`) |

## External inputs

The scripts read these public data sets, which are not redistributed here:
- the SNUMAT hybrid-functional band-gap database (Kim et al., Sci. Data 2020, 7, 387);
- Materials Project band gaps, with the gaps computed without a Hubbard U mapped to the HSE06 level by the correction model of Chen et al. and the GGA+U gaps kept as computed (Jain et al., APL Mater. 2013, 1, 011002, and Chen et al., Phys. Rev. B 2024, 109, 085306);
- the experimental gaps compiled with Roost and the Roost release code with its Matscholar embedding (Goodall and Lee, Nat. Commun. 2020, 11, 6280);
- the Matbench `expt_gap` task (Dunn et al., npj Comput. Mater. 2020, 6, 138).

## Running the scripts

The scripts keep the absolute paths of the original runs, so that every number can be traced to the file it came from (`tools/paths_original.json`). One command sets them up for a new machine:

```bash
python tools/setup_paths.py
```

It builds `work/` with the directory layout of the original runs, linking the code folders of this repository into place, creates `external/` with one folder per input, and rewrites every original path to its local location. Running it again changes nothing. Then place the public inputs as listed in `external/README.md` and run each script from any directory with the Python environment that ran the setup.

## Environment

The original runs used Python 3.10 with PyTorch 2.4 and CUDA 12.1 for training, and NumPy, pandas, SciPy, scikit-learn and pymatgen for the analysis. Server jobs were run with `OMP_NUM_THREADS=1`.

## Licence

The code is released under the MIT License (`LICENSE`). The data in `data/`, Supplementary Data 1 to 3 of the paper, are released under CC BY 4.0 (`data/LICENSE`). Please cite the paper when you use them.
