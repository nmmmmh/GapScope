# Boundaries of the results

This file lists where the results of the paper hold and where they do not. It is meant for readers who reproduce them. Each item gives the numbers and the script that produces them. Scripts named without a folder are in `analysis/refcheck/`; the repeated fold assignments are in `cv/repeated/`.

## Data

- The traced set has 237 reports of 85 compositions. Of the 40 single-phase compositions, 29 have one report, so their floor is zero by construction. They enter the composition-weighted mean of 0.080 eV.
- Three candidate papers without a retrievable full text were set aside during collection.
- The extraction used DeepSeek, version 4.1 Flash. The extraction prompts were not retained. Every record was resolved by the authors against its source, and the records are in Supplementary Data 1.
- The six further sources were collected without favouring polymorphs. Their 29 compositions have more than one report in only 3 cases and carry a floor of 0.023 eV. Under composition holdout the MAE there is 0.153 eV for Roost and 0.211 eV for the distributional model. (`analysis/refcheck/k1_conformal.py` → `k1_out.txt`)

## Where the distributional model is not ahead

- On the 47 compositions kept by the 0.1 eV agreement rule, where the reports agree, the space group has nothing to separate. There the distributional model has an MAE of 0.304 eV, against 0.270 eV for Roost, under composition holdout in the original fold assignment (SI Table S4).
- Over five fold assignments it is first on the kept compositions in one, and trails the best reference model by 0.01 to 0.04 eV in the other four.
- Its paired CRPS difference from Roost-cal under chemical-family holdout is −0.007 ± 0.004 eV. The 95% interval excludes zero in three of the five assignments.

## Calibration

- Marginal coverage on the traced reports under chemical-family holdout is about 80% for every out-of-fold scaling of the Roost scale, as it is for the distributional model:
  - distributional model 0.792;
  - Roost + conformal 0.797;
  - Roost-cal* 0.806;
  - Roost + Mondrian (D_c) 0.799.
  (`analysis/refcheck/k2_conformal_full.py`, `analysis/refcheck/k4_mondrian.py`)
- The difference lies in how the coverage is shared across compositions (SI Table S13, lower block).
- Under composition holdout, where the training data contain the same chemistry, every model shows the same gradient across the thirds of the floor:
  - distributional model 0.93 / 0.74 / 0.66;
  - Roost + conformal 0.93 / 0.83 / 0.66.
  The even coverage of the distributional model appears on new chemistry.
- Mondrian conformal prediction grouped by D_c (D_c = 0, 0 < D_c < 0.3 eV, D_c ≥ 0.3 eV) covers the flagged compositions at 0.80 under chemical-family holdout, with intervals 2.55 eV wide, against 1.88 eV for the distributional model. Under composition holdout it covers them at 0.77, against 0.61 for the distributional model. (`analysis/refcheck/k4_mondrian.py` → `k4_out.txt`)
- In the windows, Roost + conformal is close to calibrated under composition holdout: photovoltaic promised 0.72, delivered 0.73. (`analysis/refcheck/k3b_window.py`)
- Among compositions whose predicted median lies within 0.4 eV of a window limit, neither the predicted width nor D_c separates those with measured values inside and outside the window (AUC 0.46 to 0.55).

## Hybrid functional and D_c

- Split at D_c = 0.3 eV, the phase gain of HSE06 is:
  - 0.016 eV (−0.150 to 0.238) over 39 reports;
  - 0.116 eV (−0.057 to 0.241) over 29 reports.
  The intervals overlap. The gain follows the spread of all computed polymorph gaps (ρ = 0.34), not D_c (ρ = 0.11). (`analysis/refcheck/b9_split_dc.py` → `b9_out.txt`)
- The computed floor counts only polymorphs present in the computed database. A measured phase absent from it adds to the measured floor and not to the computed one. This holds for the traced compositions and for the Matbench estimate. (`analysis/refcheck/e6b_matbench_floor_nonoise.py`)
- In the benchmark of Borlido et al.:
  - the compositions with D_c ≥ 0.3 eV are nearly six times as common as across SNUMAT, and most of this enrichment comes from having more recorded polymorphs (64% against 44% among compositions with two or more accessible polymorphs);
  - sixteen compositions already enter the benchmark more than once with different structures, SiC four times and TlI three times among them;
  - every material is computed on its matched experimental structure.
