# Error budget for comparing computed and experimental band gaps

Evaluation only: no training, no GPU, no new DFT. Everything here is derived from already-calculated
SNUMAT HSE/GGA gaps and from the two experimental sets. All scripts live in this directory and were run with
`/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python -I`.

| script | what it does | outputs |
|---|---|---|
| `p1_snumat.py` | parses `snumat.json` into a tidy table; establishes the relation to `hse_data.csv` | `snumat_tidy.csv`, `hse_csv_keyed.csv` |
| `b1_match.py` | item 1 (optical vs fundamental) | `item1_optical_vs_fundamental.csv`, `fig_csv/fig_item1_deltas.csv` |
| `b1b_match.py` | composition matching, all counts | `match_counts.json`, `matched_keys.csv` |
| `b2_poly.py` | item 2, polymorph spread vs experimental spread | `snumat_by_comp_sg.csv`, `snumat_spread_by_composition.csv`, `item2_rank_correlation.csv`, `fig_csv/fig_item2_spread_by_composition.csv` |
| `b2c_choice.py` | item 2b, ground state vs space-group-matched polymorph | `item2b_sg_matched_records.csv`, `item2b_choice_effect.csv` |
| `b3_var.py` | item 3, phase / source / specimen spread in eV | `item3_variance_components_ev.csv`, `fig_csv/fig_item3_per_composition_exp.csv` |
| `b4_budget.py` | item 4, the budget | `budget.csv`, `per_record.csv`, `fig_csv/fig_item4_budget_terms.csv` |
| `b5b_percrps.py` | per-record CRPS/PIT of the computed values | `fig_csv/fig_item5_crps_per_record.csv` |
| `b5_verdict.py` | items 5 and 6 | `item5_verdict.csv`, `item5c_crps.csv`, `item5c_view_contrast.csv`, `item6_conventions.csv` |

## 1. Data sources and the relation between the two SNUMAT files

`/home/czx/2024/doped/data/snumat.json` — 10,481 records, each with a unique `SNUMAT_id`, carrying
`Band_gap_HSE`, `Band_gap_HSE_optical`, `Band_gap_GGA`, `Band_gap_GGA_optical`, both direct/indirect flags,
the `SOC` flag, `Space_group_rlx`, `Magnetic_ordering` and the relaxed structure. Composition is taken from
`atoms['elements']` (no formula field exists in the file). 8,730 distinct compositions; 10,186 distinct
(composition, relaxed space group) cells. SOC is on for 129 records; magnetic ordering is NM for 7,840.

`/home/czx/2024/spinels/data/hse_data.csv` — 13,719 rows (13,718 with a parseable formula), 10,739 distinct
compositions, columns `Material, Chemical_formula, Band_gap_HSE, Direct_or_indirect_HSE, Band_gap_GGA,
Space_group_cif, Free_energy`.

**The csv is neither a subset nor a reshaping of the json. They are two overlapping SNUMAT exports.**

* 6,107 records match exactly on (canonical composition key, space group, HSE, GGA) — identical numbers.
* Matching on (composition key, HSE value) alone gives 6,646 records of 10,481 / 13,718.
* 6,993 json records sit on a composition that occurs somewhere in the csv; 3,488 json records are on
  compositions absent from the csv altogether. 347 json records share a composition with the csv but no HSE value.
* Where a composition+space-group cell is unique in both files, HSE agrees to <1e-4 eV for 99.4% of 5,614 cells
  and GGA for 99.3%; the median absolute difference is exactly 0.
* The space-group fields are not the same quantity: the json reports the relaxed-structure space group
  (`Space_group_rlx`), the csv the input-CIF space group (`Space_group_cif`). On records uniquely identified by
  `Material` in both files they agree for 92.4%.

**Decision.** The json is the primary source, because it alone carries the optical gaps and the SOC flag.
`Free_energy` is not in the json, so for the ground-state-polymorph question it is joined from the csv on the exact
(composition key, space group, HSE, GGA) quadruple. That covers 6,107 of 10,481 SNUMAT records (58.3%);
coverage is stated wherever this is used, and no composition is dropped for lack of it elsewhere.

### Experimental sets

* **Provenance-linked.** `suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv`: **237 records,
  85 compositions**, 205 specimen groups, 70 source groups; 234 carry a `phase_label`, 90 a reported
  `space_group`. The copy in `polymorph/` is a 205-record / 56-composition subset of it (every `record_id` in the
  small file appears in the large one) and is the one the earlier `within_phase*.py` scripts used. This analysis
  uses the 237-record file; the brief's "237 records / 56 compositions" mixes the two.
* **Compilation.** `cv/data/tierB_rows.csv`: 3,728 rows, 2,467 compositions.
* 117 provenance records (27 compositions) sit on a composition that also appears in the compilation set.

### Canonical composition key and the matching rule

`'|'.join(f'{el}:{frac[el]:.12f}' for el in sorted(elements))` on the **fractional** composition
(`pymatgen.Composition.fractional_composition`), identical to `poly_corr.py` and to the published
`composition_key`. It reproduces the published key for 234 of 237 provenance records; the three exceptions are the
MAPbBr3 rows, where the published key is built from the organic cation written out (C/H/N explicit) and pymatgen
parses `MAPbBr3` differently. Those rows carry their published key, so nothing is lost.

Matching is **composition-level only** — there is no shared identifier between SNUMAT and the experimental
records. A second, stricter level matches a record to the SNUMAT cell whose relaxed space group equals the
record's own space group (reported where available, otherwise the `model_space_group_number` from
`records/metadata.json`; 90 reported, 145 with either).

**Counts I obtain from the json:**

| set | compositions | matched to SNUMAT |
|---|---|---|
| provenance (237-record file) | 85 | **56** |
| provenance (205-record file, for comparison with the existing table) | 56 | **50** |
| compilation (tier B) | 2,467 | **790** |
| union | 2,522 | **817** |

Of the 817 overlapping compositions, **231 (28.3%) carry more than one SNUMAT space group**; among the 56
matched provenance compositions, 35 do. 196 of the 237 provenance records and 1,778 of the 3,728 compilation
rows sit on a matched composition.

The existing `cv/snumat_overlap.csv` has 1,452 rows over 682 compositions, of which 618 are in my json-based
match set. It was built from the csv export, so it is larger in compositions (the csv covers more compositions)
and smaller in evidence (no optical gaps, no SOC flag). The brief's figures (44/56, 665/2,467, 247/682) are that
table's; mine are the json's.

When one composition+space-group cell holds several SNUMAT records (different ICSD entries), the cell value is
the **median** over those records. This is the only aggregation inside a cell.

## 2. Item 1 — optical versus fundamental gap

`item1_optical_vs_fundamental.csv`. The difference `Band_gap_*_optical − Band_gap_*` is never negative
(0 records of 10,481 for either functional).

HSE, all 10,481 records: mean **0.679 eV**, median **0.297 eV**, Q1 0.000, Q3 0.916, p90 1.863,
**64.3%** above 0.1 eV, 49.8% above 0.3 eV, 39.2% above 0.5 eV, max 14.41.

**The brief's measured numbers are confirmed exactly** (median 0.297, mean 0.679, 64.3% above 0.1 eV).

The dependence on direct/indirect character is the whole story: restricted to records with a non-zero HSE gap,
**direct** materials (n=2,952) have median 0.000 and mean 0.0001 eV — 2,951 of 2,952 are exactly zero —
while **indirect** materials (n=7,528) have median **0.577**, mean **0.946**, Q1 0.237, Q3 1.238, p90 2.268, and
89.6% exceed 0.1 eV. GGA behaves the same way: all 10,481 records give mean 0.633, median 0.276, 65.1% above
0.1 eV; direct records are identically zero, indirect ones have median 0.502, mean 0.849.
Masking SOC records or restricting to non-magnetic systems moves nothing (HSE median 0.300 / 0.333).

**Which quantity a measurement corresponds to.** SNUMAT's "optical" gap is the lowest *dipole-allowed direct*
transition; its "fundamental" gap is the band extrema difference. A Tauc analysis plotted as
(αhν)² against hν, ellipsometry, and thermoreflectance all extract a direct allowed transition, so they
correspond to the **optical** gap. An indirect Tauc plot ((αhν)^(1/2)), and a bare absorption-edge or
diffuse-reflectance onset, correspond to the **fundamental** gap. In the provenance set the split is
151 records probing the direct optical gap (129 direct-allowed Tauc, 14 ellipsometry, 8 thermoreflectance) and
86 the fundamental gap (44 indirect Tauc, 27 absorption edge, 1 PL, 14 unclassified). This matters because the
two definitions differ by **zero** for a direct material and by **1.3 eV (median)** for an indirect one —
on the 96 space-group-matched records the optical-minus-fundamental difference is median 0.000 / mean 0.025 eV
for the 38 HSE-direct cells and median **1.309** / mean **1.677** eV for the 58 HSE-indirect cells.
Neither definition is "the" experimental gap, and the choice is worth more than the functional.

## 3. Item 2 — polymorph choice

`snumat_spread_by_composition.csv`, `item2_rank_correlation.csv`, `fig_csv/fig_item2_spread_by_composition.csv`.

Over the 929 SNUMAT compositions with ≥2 space groups **and** at least one experimental report, the HSE spread
across polymorphs of one composition is: max−min mean 0.542, **median 0.262**, Q1 0.056, Q3 0.704, p90 1.453 eV;
standard deviation across polymorphs median 0.167 eV. GGA is slightly tighter (range median 0.203, SD median 0.123).

**Composition by composition, the computed spread does not track the experimental spread.** Spearman rank
correlation between the HSE polymorph range and the experimental range:

| set | n | ρ (HSE range) | p | ρ (GGA range) | p |
|---|---|---|---|---|---|
| provenance | 32 | 0.202 | 0.27 | 0.316 | 0.078 |
| compilation | 123 | 0.076 | 0.40 | 0.081 | 0.37 |
| both | 142 | 0.143 | 0.091 | 0.178 | 0.034 |

On standard deviations instead of ranges the correlations are weaker still (ρ ≤ 0.31, all p > 0.08). This is a
**null result**: the size of the computed polymorph spread is not a usable predictor of how widely a composition's
measurements will scatter. The one signal that does survive is cruder — compositions with ≥2 SNUMAT space groups
have a larger median experimental range than single-space-group compositions (provenance 0.325 vs 0.125 eV;
both tiers 0.280 vs 0.200 eV).

### Ground state versus the space-group-matched polymorph

`item2b_choice_effect.csv`. 96 provenance records (36 compositions) match a SNUMAT cell with the same space
group; 84 of them (31 compositions) also have a `Free_energy` ground state. The matched cell *is* the
lowest-`Free_energy` cell for 51.2% of them.

| reference polymorph | HSE MAE | GGA MAE |
|---|---|---|
| space-group matched | 0.534 | 1.298 |
| lowest `Free_energy` | 0.597 | 1.269 |
| median over the composition's polymorphs | 0.553 | 1.290 |

The aggregate MAE barely moves (+0.06 eV for HSE, −0.03 for GGA). **Individual records move much more.** On the
41 records where the ground-state cell is not the matched cell, the HSE MAE goes 0.616 → 0.747 eV and the median
absolute change in a single record's error is **0.240 eV** (GGA 0.386 eV); the 90th percentile of the change over
all 84 records is 0.59 eV (HSE) / 0.60 eV (GGA). Averaging hides the polymorph-choice error; it does not remove it.

## 4. Item 3 — specimen and source spread, in eV

`item3_variance_components_ev.csv`. Computed on the 237-record provenance set with one-way random-effects moment
estimates pooled over cells, so each number is a standard deviation in eV.

| level | grouping | cells | reports | between (eV) | within (eV) | raw variance share |
|---|---|---|---|---|---|---|
| within composition | phase | 56 | 208 | **0.231** | **0.100** | 0.90 |
| within phase | source group | 34 | 112 | 0.073 | 0.095 | 0.13 |
| within phase | specimen group | 34 | 112 | **0.080** | **0.066** | 0.85 |
| within phase | measurement method | 34 | 112 | 0.062 | 0.092 | 0.28 |
| within phase | gap definition | 34 | 112 | 0.064 | 0.096 | 0.14 |

Total within-composition SD **0.227 eV** (208 reports, 56 compositions with ≥2 reports); total
within-(composition, phase) SD **0.0995 eV** (112 reports in 34 cells). Pooled within-specimen SD **0.066 eV**
over the 18 specimen cells that have ≥2 reports.

Re-running the existing `polymorph/within_phase.py` and `within_phase2.py` unchanged on their 205-record file
reproduces their published picture: on the 13 multi-phase compositions the within-phase sum of squares is 15.5%
of the within-composition sum of squares, so **phase accounts for about 85%**, not three quarters; within a phase,
`specimen_group_id` has raw share 0.83 (adjusted 0.48) while `source_group_id` has 0.14 (0.09) and the method
category 0.00. Temperature explains none of it — no cell in that set has more than one recorded temperature.
My 237-record version gives the phase share as 0.90. **The brief's "about 3/4" understates it; the correct
statement is 85–90%, and the "within a phase it is mostly specimen" half is confirmed.**

In budget units: choosing the wrong phase costs **0.23 eV**; inside the right phase, changing specimen costs
**0.08 eV** and everything else (source, method, gap definition, repeats) **0.07 eV**.

## 5. Item 4 — the budget

`budget.csv`, `per_record.csv` (196 rows), `fig_csv/fig_item4_budget_terms.csv`.

196 provenance records on 56 SNUMAT-matched compositions; 96 of them (36 compositions) also space-group matched.
The computed reference is the space-group-matched cell where one exists, otherwise the median over the
composition's polymorphs. Terms are magnitudes in eV, averaged over the 196 records.

**These terms are not orthogonal and must not be added.** The optical/fundamental term and the polymorph term
are strongly coupled: whether a cell is direct or indirect is a property of the polymorph, so changing polymorph
can change the optical term by more than 1 eV. The specimen/source term and the polymorph term overlap too:
part of the reason two specimens disagree is that they are different phases. Hence a residual *range*, not a
subtraction.

| term | n | mean (eV) | median | Q1 | Q3 | p90 |
|---|---|---|---|---|---|---|
| **HSE** total \|computed fundamental − experiment\| | 196 | **0.592** | 0.504 | 0.310 | 0.825 | 1.193 |
| optical/fundamental term | 196 | 0.701 | 0.000 | 0.000 | 0.947 | 3.311 |
| total after definition matching | 196 | 0.893 | 0.578 | 0.342 | 1.165 | 2.299 |
| polymorph-choice term | 196 | 0.777 | 0.463 | 0.000 | 0.783 | 1.416 |
| specimen/source term | 196 | 0.171 | 0.097 | 0.064 | 0.281 | 0.337 |
| residual, upper bound (nothing assigned) | 196 | **0.592** | 0.504 | | | |
| residual, mid (definition + space group matched, experiment as its interval) | 196 | **0.701** | 0.353 | | | |
| residual, lower bound (best polymorph × best definition vs the interval) | 196 | **0.138** | 0.000 | 0.000 | 0.170 | 0.358 |
| **GGA** total | 196 | **1.216** | 1.115 | 0.600 | 1.543 | 2.417 |
| optical/fundamental term | 196 | 0.582 | 0.000 | 0.000 | 0.520 | 2.259 |
| total after definition matching | 196 | 0.845 | 0.748 | 0.414 | 1.224 | 1.521 |
| polymorph-choice term | 196 | 0.735 | 0.479 | 0.000 | 0.727 | 1.320 |
| specimen/source term | 196 | 0.171 | 0.097 | 0.064 | 0.281 | 0.337 |
| residual, upper bound | 196 | **1.216** | 1.115 | | | |
| residual, mid | 196 | **0.655** | 0.574 | | | |
| residual, lower bound | 196 | **0.204** | 0.000 | 0.000 | 0.094 | 0.668 |

Term definitions: the optical term is `optical − fundamental` on the matched cell, set to zero for the
75 of these 196 records whose analysis targets the fundamental gap (121 probe the direct optical gap; the split
over all 237 provenance records is 151 / 86). The polymorph term is half the range of the *probed* quantity across
the composition's SNUMAT space groups; the specimen/source term is the experimental SD within the composition
(zero where there is only one report). The mid residual is the distance from the definition-matched,
space-group-matched computed value to the interval [min report, max report]; the lower bound is the distance
from the closest value attainable by any (polymorph, gap definition) combination to that same interval.

**The mid assignment is worse than the raw total for HSE (0.701 vs 0.592) and better for GGA (0.655 vs 1.216).**
That is not a bookkeeping slip, it is the result: HSE's apparent agreement with experiment partly rests on
comparing a fundamental gap with an optical measurement, and the two errors cancel. Applying the optical
correction honestly pushes HSE *above* experiment — the bias at the raw total is −0.14 eV, so adding 0.70 eV of
optical correction overshoots. GGA's bias is −1.16 eV, so the same correction helps it.

## 6. Item 5 — does the verdict change?

`item5_verdict.csv`, `item5c_crps.csv`, `item5c_view_contrast.csv`.

| reference | computed value | n | MAE (eV) | bias (eV) |
|---|---|---|---|---|
| (a) composition median, common practice | HSE, polymorph median | 56 | **0.538** | −0.081 |
| | GGA, polymorph median | 56 | **1.079** | −1.027 |
| | HSE, lowest `Free_energy` | 46 | 0.544 | −0.066 |
| | GGA, lowest `Free_energy` | 46 | 1.056 | −1.031 |
| (b) space-group-matched experimental record | HSE, matched cell | 96 | **0.590** | −0.077 |
| | GGA, matched cell | 96 | **1.194** | −1.106 |
| (b) same 96 records, polymorph median instead | HSE | 96 | 0.590 | −0.105 |
| | GGA | 96 | 1.183 | −1.122 |
| (b2) every matched report, no median | HSE | 196 | 0.592 | −0.138 |
| | GGA | 196 | 1.216 | −1.158 |

(c) CRPS of the computed value under the model's out-of-fold predictive distribution for the same composition.
The main model's calibrated distribution is `EASYUQ_ON_GS_SELCAL_PRE_NZ` (IDR and kernel-smoothed), a weighted
Laplace mixture per tier-A record; CRPS and PIT are evaluated on the 187 matched records that are in tier A.
For reference, the same distributions scored at the *experimental* value are given alongside.

| distribution | quantity scored | n | CRPS | median PIT | PIT in (0.1, 0.9) |
|---|---|---|---|---|---|
| EasyUQ-KS (main) | HSE computed | 187 | 0.471 | 0.573 | 61.0% |
| | GGA computed | 187 | 0.883 | 0.039 | 36.4% |
| | experiment | 187 | 0.392 | — | — |
| EasyUQ-IDR | HSE computed | 187 | 0.480 | 0.618 | 59.9% |
| | GGA computed | 187 | 0.902 | 0.029 | 25.1% |
| raw GS mixture, composition only | HSE computed | 187 | **0.4626** | — | 83.4% |
| | GGA computed | 187 | 0.6554 | — | 65.8% |
| | experiment | 187 | **0.4628** | — | — |
| raw GS mixture, composition + space group | HSE computed | 187 | 0.4690 | — | 78.1% |
| | GGA computed | 187 | 0.6821 | — | 64.7% |
| | experiment | 187 | 0.4525 | — | — |

**The ranking does not change: HSE beats GGA under every reference.** What changes is the *size* of the gap and
what it means. The apparent MAE moves by at most 0.05 eV for HSE (0.538 → 0.592) and 0.14 eV for GGA
(1.056 → 1.216) across (a), (b) and (b2) — the choice of experimental reference is a 10% effect on HSE and a 13%
effect on GGA, far smaller than the budget terms of item 4. Under the model's own predictive distribution the
HSE value is **statistically indistinguishable from a real experimental report** (CRPS 0.4626 vs 0.4628 for the
experiment, median PIT 0.57, 83% of records with PIT in the central 80%), whereas the GGA value sits in the far
left tail (median PIT 0.039). Conditioning the distribution on composition *and* space group changes nothing
material (HSE 0.4626 → 0.4690).

Note on the view contrast: in the calibrated `EASYUQ_ON_GS_*` arm the four views carry identical distributions by
construction, so the space-group conditioning can only be tested on the raw 3-seed GS lognormal mixtures
(`cv/gs_out/comp_GS_seed*_fold*.npz`, out-of-fold); that is what the last two rows use.

**How often the computed value is not distinguishable from experiment at all.** Of the 196 matched records, the
HSE fundamental gap falls inside [min report, max report] for its composition in **22 (11.2%)** and GGA in
6 (3.1%). Definition-matched: HSE 18 (9.2%), GGA 31 (15.8%). But if *any* combination of the composition's
SNUMAT polymorphs and the two gap definitions is allowed, **121 of 196 (61.7%) for HSE and 138 of 196 (70.4%)
for GGA** overlap the experimental interval. At composition level, 7 of the 50 compositions with ≥2 reports have
their HSE polymorph median inside the reported interval, and 2 have GGA.

## 7. Item 6 — what the cleaning conventions do

`item6_conventions.csv`. Computed reference is the polymorph median per composition.

| set | functional | reports | compositions | MAE, every report | MAE, composition median | Δ | compositions with span ≤0.1 eV kept | removed | MAE after discard | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| provenance | HSE | 196 | 56 | 0.593 | 0.538 | **−0.054** | 20 | **36** | 0.513 | −0.026 |
| provenance | GGA | 196 | 56 | 1.211 | 1.079 | **−0.132** | 20 | 36 | 0.892 | **−0.188** |
| compilation | HSE | 1,778 | 790 | 0.546 | 0.585 | +0.039 | 577 | 213 | 0.584 | −0.002 |
| compilation | GGA | 1,778 | 790 | 1.000 | 0.979 | −0.021 | 577 | 213 | 0.952 | −0.027 |
| both | HSE | 1,974 | 817 | 0.551 | 0.586 | +0.036 | 578 | 239 | 0.587 | +0.001 |
| both | GGA | 1,974 | 817 | 1.021 | 0.981 | −0.040 | 578 | 239 | 0.951 | −0.030 |

Both conventions flatter a functional, and they flatter the worse functional more. On the provenance set, taking
the median of several reports removes 0.054 eV of HSE error and 0.132 eV of GGA error; discarding the
36 of 56 compositions (64%) whose reports span more than 0.1 eV removes a further 0.026 eV from HSE and
0.188 eV from GGA. Together they cut GGA's apparent error by 0.32 eV (26%) and HSE's by 0.08 eV (14%), so the
apparent HSE-over-GGA advantage shrinks from 0.62 eV to 0.38 eV. On the much larger compilation set the effects
are small (≤0.04 eV) and the median convention even raises HSE's apparent error, because that set has few
multiply-reported compositions (271 of 790) and no provenance to tell phases apart.

## 8. What the data supports and what it does not

Supported:

* Optical and fundamental gaps differ by a median of 0.30 eV over all SNUMAT entries and by a median of 0.58 eV
  (mean 0.95) for indirect materials; for direct materials they are identical. 94% of the provenance records are
  optical-onset measurements, and 151 of 237 target the direct allowed transition.
* Polymorph choice is worth a median 0.26 eV of HSE gap across SNUMAT space groups of one composition, and
  changing a single record's reference polymorph from the space-group-matched cell to the ground state changes
  its error by a median 0.24 eV where the two differ.
* Phase identity carries 85–90% of the within-composition experimental variance, 0.23 eV as a standard
  deviation; specimen within a phase carries 0.08 eV and everything else 0.07 eV.
* Under the loosest defensible assignment, 62% (HSE) and 70% (GGA) of matched provenance records can be made to
  overlap the experimental spread of their composition by choosing the polymorph and the gap definition.
  The functional residual for HSE is therefore bounded between 0.14 and 0.59 eV, and for GGA between 0.20 and
  1.22 eV.
* HSE outranks GGA under every reference tried, and the ranking is robust to the choice of experimental
  reference (≤0.14 eV effect). Under the model's own out-of-fold predictive distribution the HSE value is
  indistinguishable from an experimental report (CRPS 0.4626 vs 0.4628) and the GGA value is not (median PIT 0.04).
* The two common cleaning conventions reduce a functional's apparent error, and they reduce GGA's more than
  HSE's: on the provenance set together they shrink the HSE-over-GGA advantage from 0.62 to 0.38 eV, while
  discarding 64% of the compositions.

Not supported:

* **That the computed polymorph spread predicts the experimental spread of the same composition.** All rank
  correlations are ≤0.32 and only one of twelve reaches p < 0.05 after the obvious multiplicity. The honest
  statement is that both spreads are large and that they are not related composition by composition.
* **That the budget terms can be added or subtracted to isolate a functional error.** The terms overlap by
  construction; only the bounded range is defensible. The point estimates in `budget.csv` are magnitudes, not
  an additive decomposition.
* **That applying the optical correction improves HSE.** It does not: HSE's raw bias is −0.14 eV and the
  correction averages +0.70 eV on the records it applies to, so definition matching raises HSE's apparent error
  from 0.59 to 0.89 eV. Any claim that HSE "agrees with experiment to ~0.5 eV" rests on comparing a fundamental
  gap with an optical measurement.
* **That the aggregate MAE is insensitive to polymorph choice.** The MAE is (±0.06 eV), the individual records
  are not (median change 0.24 eV, p90 0.59 eV). A benchmark that reports only an MAE cannot see this.
* **That space-group conditioning sharpens the comparison.** Adding the space-group view to the model's
  predictive distribution changes the CRPS of the computed value by 0.006 eV, and matching the record's space
  group rather than taking the polymorph median changes the MAE by 0.001 eV (HSE) and 0.011 eV (GGA) on the
  same 96 records.
* Nothing here is a temperature analysis: no cell in the provenance set has more than one recorded temperature,
  so the thermal contribution to the experimental spread cannot be separated from specimen identity.
* The `Free_energy` ground state is available for 58% of SNUMAT records and 31 of the 36 space-group-matched
  compositions, so the ground-state comparison in §3 rests on 84 records; no hull energy was recomputed.
