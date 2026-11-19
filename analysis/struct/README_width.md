# What chemistry sets how wide a composition's experimental band-gap distribution is

Server-only, under `/data/czx/GapScope_Revision_20261113/struct/`, with
`/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python -I`. Scripts: `width.py`
(targets, predictors, models, categorical contrasts) and `width2.py` (robustness, the coarse rule,
and the CRPS value of a predicted width). Outputs: `width.csv`, `tables/width_univariate.csv`,
`tables/width_models.csv`, `tables/width_categorical.csv`, `tables/width_importance_*.csv`,
`tables/width_crps.csv`. Structural predictors come from `descriptors.csv` (stage 1).

## 1. The target

Per composition key, over the reported experimental values:
**`exp_range`** = max − min, **`exp_sd`** = sample standard deviation (ddof 1), **`exp_iqr`** =
P75 − P25, plus `exp_median` and `n_reports`. Computed separately on the two sets, never pooled.

| set | compositions | with ≥ 2 reports | of those, matched to SNUMAT |
|---|---|---|---|
| provenance (237-record Supplementary Data 1) | 85 | 56 | 50 |
| compilation (`cv/data/tierB_rows.csv`, 3,728 rows) | 2,467 | 452 | 271 |

**Compositions with a single report are excluded from every fit and every correlation.** Their
spread is not measured as zero, it is unmeasured, and treating it as zero is what makes the
"polymorph count predicts width" effect look far stronger than it is (see section 2). They are
kept in `width.csv` with `exp_range = NaN` and are reported separately where it matters.

## 2. The initial claim, and what the sample size does to it

Reproduced on the provenance set, SNUMAT-matched compositions:

| treatment | 1 space group | ≥ 2 space groups | Mann–Whitney p |
|---|---|---|---|
| single-report compositions counted as range 0 | median 0.080 eV (n=21) | 0.280 eV (n=35) | 0.014 |
| only compositions with ≥ 2 reports | **0.110 eV (n=17)** | **0.320 eV (n=33)** | **0.046** |

The quoted 0.125 vs 0.325 eV is the second row, and it survives: compositions with more than one
accessible SNUMAT space group do have about three times the reported spread. On the **compilation**
set the same contrast is 0.200 vs 0.250 eV (p = 0.074) — present but much weaker.

The confound that has to be removed first is **how many times a composition was measured**:
Spearman(`exp_range`, `n_reports`) = **+0.509** on the compilation set and **+0.241** on the
provenance set; for `exp_sd` it is +0.222 and +0.032. Every correlation below is therefore also
reported as a partial Spearman controlling for `n_reports`.

## 3. Predictors

*From SNUMAT, per composition* (over one representative structure per relaxed space group, the
record whose HSE gap is closest to the cell median): `n_polymorphs` (distinct relaxed space
groups), `n_snumat` (records), `n_crystal_systems`, `any_noncentro`, `frac_direct`,
`mixed_direct_indirect`; the **spread** (`spr_*`, max − min) and the standard deviation (`sd_*`)
and the mean (`mean_*`) across those polymorphs of cation coordination number (mean and max), mean
coordination of all sites, mean shared vertices per polyhedral connection, corner/edge/face
fractions, bonded-network dimensionality, mean cation–anion bond length, distortion index, volume
per atom, packing fraction, number of symmetry operations, isolated-polyhedron fraction; the
computed gap spreads `gapspr_hse`, `gapsd_hse`, `gapspr_hseopt`, `gapspr_gga`; and the gap level
`hse_mean`. Three coarse flags: `var_coordination` (spread of mean cation CN ≥ 0.5),
`var_dimensionality` (dimensionality spread ≥ 1), `var_connectivity` (spread of face- or
edge-sharing fraction ≥ 0.2), and `multi_polymorph` (`n_polymorphs` > 1).

*From the composition alone* (available for every composition, SNUMAT match or not):
`n_elements`, `anion_frac`, mean anion and cation electronegativity, `en_diff`, `ionicity`,
`en_std_all`, `nval_per_atom`, `radius_ratio`, `mean_atomic_radius`, the dominant anion's Z, and
the class flags `is_halide`, `is_oxide`, `is_chalcogenide`, `is_pnictide`, `is_intermetallic`.
Cation/anion assignment and the chemistry definitions are the stage-1 ones (`README.md` section 2).

Everything is cross-validated with 5-fold `GroupKFold` on the **element system** (the set of
elements), i.e. whole chemistries held out.

## 4. Results

### Provenance set (n = 50 compositions, 49 element systems) — the hypothesis holds

Univariate (`tables/width_univariate.csv`), target `exp_range`:

| predictor | Spearman | p | partial on n_reports |
|---|---|---|---|
| spread of mean cation CN | **+0.413** | 0.0035 | **+0.395** |
| spread of distortion index | +0.410 | 0.0038 | +0.409 |
| `var_coordination` flag | +0.407 | 0.0034 | +0.400 |
| spread of mean shared vertices | +0.363 | 0.0095 | +0.373 |
| spread of mean bond length | +0.337 | 0.019 | +0.371 |
| computed GGA polymorph gap spread | +0.334 | 0.018 | +0.368 |
| `var_connectivity` flag | +0.334 | 0.018 | +0.343 |
| `multi_polymorph` | +0.287 | 0.043 | +0.323 |
| computed HSE polymorph gap spread | +0.284 | 0.045 | +0.337 |
| spread of dimensionality | +0.231 | 0.106 | +0.201 |
| number of reports | +0.339 | 0.016 | 0 |

The structural-freedom predictors are **stronger than the computed gap spread** and they are not
the sample-size effect: the partial correlations are unchanged.

Models, whole element systems held out (`tables/width_models.csv`), target `exp_range`:

| feature set | model | Spearman | R² |
|---|---|---|---|
| `n_reports` only | ridge | 0.144 | 0.056 |
| composition chemistry only | gbm | 0.474 | −0.251 |
| composition chemistry only | ridge | 0.371 | 0.029 |
| **chemistry + SNUMAT structure, no computed gap spread** | **gbm** | **0.590** | **0.167** |
| chemistry + structure + n_reports | gbm | 0.557 | 0.134 |
| everything including the computed gap spread | gbm | 0.485 | 0.040 |
| computed gap spread only | gbm | 0.220 | −0.208 |

**The relation survives removal of the computed polymorph gap spread — it is in fact best without
it.** Target `exp_sd` (free of the sample-size effect) behaves the same way: 0.424 for
chemistry + structure, 0.382 for chemistry alone, −0.405 for `n_reports` alone.

Robustness (`width2.py`): the held-out Spearman is **+0.586**, bootstrap 95% CI over compositions
**[0.393, 0.722]**, and a label-permutation null over the same folds gives mean −0.026, 95th
percentile +0.263, **p < 0.005**. With n = 50 compositions this is a real but loosely determined
number.

Permutation importance of the full model (eV of MAE, out of fold): `gapspr_gga` 0.0044,
`en_cat_mean` 0.0028, `hse_mean` 0.0023, `spr_mean_shared_vertices` 0.0020, `spr_cn_all_mean`
0.0020, `mean_distortion_index` 0.0016, `frac_direct` 0.0016. No single predictor is worth 5 meV;
the ranking signal is spread thinly over the coordination/connectivity spreads plus the
electronegativity of the cation sublattice and the gap level.

### Compilation set (n = 271 matched, 452 total) — the hypothesis fails

Nothing beats the number of reports. Best Spearman: `n_reports` alone 0.483; chemistry + n_reports
0.461; chemistry + structure + n_reports 0.483; chemistry alone 0.139–0.152; computed gap spread
alone −0.099. **After partialling out `n_reports` every predictor collapses to |ρ| ≤ 0.06**
(largest: spread of mean shared vertices +0.051, `var_coordination` +0.061). On the n-insensitive
`exp_sd` the best held-out Spearman is 0.258. The compilation set's spread is a record of how often
and how carelessly a composition has been measured, not of its structural freedom.

### Negative control — confirmed and sharpened

The earlier finding stands: the composition's **computed** polymorph gap spread does not track its
experimental spread (provenance ρ = 0.284, p = 0.045; compilation ρ = 0.121, p = 0.047, partial on
n = −0.041). The coarser, categorical predictors do work where it failed, on the provenance set
(`tables/width_categorical.csv`, `width2.py`):

| coarse rule | median range, flag 0 | median range, flag 1 | difference (95% CI) | ratio | p |
|---|---|---|---|---|---|
| variable cation coordination | 0.140 eV (n=33) | **0.410 eV (n=17)** | +0.270 [0.110, 0.610] | 2.9× | 0.0045 |
| variable connectivity | 0.140 eV (n=33) | 0.410 eV (n=17) | +0.270 [0.040, 0.400] | 2.9× | 0.020 |
| more than one polymorph | 0.110 eV (n=17) | 0.320 eV (n=33) | +0.210 [0.000, 0.330] | 2.9× | 0.046 |
| variable dimensionality | 0.160 eV (n=42) | 0.445 eV (n=8) | — | 2.8× | 0.115 |

Two chemistry-class effects also appear on the provenance set and should be treated with care at
this sample size: oxides are wider (median 0.375 vs 0.110 eV, p = 0.011) and halides narrower
(0.040 vs 0.265 eV, p = 0.019); neither replicates on the compilation set (0.200 vs 0.220 and
0.280 vs 0.200).

## 5. Could a chemistry-predicted width beat a freely fitted one, and by how much

Two separate answers, and they point in different directions.

**Ranking: yes.** A width predicted from chemistry and SNUMAT structural freedom ranks the observed
experimental spread with held-out Spearman **0.586 [0.393, 0.722]** on the provenance compositions,
while a freely fitted global width ranks them at **0 by construction**. Note that this is *not* the
same quantity as the 0.477 currently quoted for the model (predicted width against absolute median
error on tier A): mine is against the measured spread of the reports themselves. The two can be
compared only after stage 2 puts the predictor inside the model, so I am not claiming that 0.586
replaces 0.477.

**Sharpness: the expected gain is about 1%.** Scoring a zero-centred normal against the 190
provenance records' deviations from their composition median, out of fold with element systems held
out (`tables/width_crps.csv`):

| width | mean CRPS (eV) | gain over the constant |
|---|---|---|
| one global constant, freely fitted | 0.1058 | — |
| two-group rule (variable coordination) | 0.1049 | 0.8% |
| chemistry + structure model | 0.1045 | **1.2%** |

That is the honest expected size of the effect **before** anything is retrained: a 2.9× spread in
the *right* direction on a third of the compositions buys around one percent of CRPS, because CRPS
is dominated by the central mass of the distribution and most compositions are narrow either way.
Two caveats, both pointing to the real gain being different rather than larger: this proxy scores
only the within-composition scatter about its own median, whereas the deployed width must also
cover the model's own error (so the deployed width is larger and a given relative change in it
matters less), and n = 50 compositions with 190 records is a thin basis for a sharpness number.

## 6. Plain statement

* On the provenance set — the set the paper's distributions are about — **the hypothesis is
  supported**: how much the cation coordination number, the polyhedral connectivity and (weakly)
  the dimensionality vary among a composition's accessible polymorphs predicts how widely its
  experimental gap has been reported, Spearman ≈ 0.59 out of fold with whole chemistries held out,
  and it is not an artefact of how many times the composition was measured.
* The **coarse** form of the predictor is the robust one. A single binary flag — does this
  chemistry allow polymorphs with different cation coordination — separates median reported ranges
  of 0.14 and 0.41 eV. The fine-grained computed polymorph gap spread remains the weak predictor it
  was, and the model is better without it.
* On the **compilation** set the hypothesis fails: there the spread is explained by the number of
  reports and by nothing else we can compute. Any width model trained on tier B spreads would learn
  reporting volume, not physics.
* The width would become **physically meaningful and genuinely informative in ranking**, which is
  what fixes the measured deficiency, but the expected **sharpness** gain is of order 1% of CRPS on
  the evidence here, not a factor. It should be sold as "the width now has a physical origin and
  orders compositions correctly", not as a large accuracy gain.
* Hard limits to state in any paper text: 50 compositions carry the whole provenance result;
  single-report compositions carry no information about width and are excluded; and 2 of the 85
  provenance compositions-with-spread (6 of the 56) have no SNUMAT match at all, so a deployed
  predictor needs the composition-only fallback, which on its own reaches Spearman 0.37–0.47.
