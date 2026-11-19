# Stage 1 — what structural and chemical features control the gap difference between polymorphs of one composition

Everything here runs on the server under `/data/czx/GapScope_Revision_20261113/struct/` with
`/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python -I`. No GPU, no new DFT, no
training of the GapScope model: stage 1 is an analysis of the SNUMAT computed gaps only.

| file | what it is |
|---|---|
| `desc_worker.py` | descriptors for one shard of SNUMAT records |
| `run_desc.sh` | launches the shards (24 processes, one BLAS thread each) |
| `merge_desc.py` | merges the shards into `descriptors.csv`, reports failures |
| `pairs_analysis.py` | cells, pairs, models, hypothesis contrasts; writes `pairs.csv` and `tables/*.csv` |
| `case_check.py` | the Ga2O3 and FAPbI3 checks |
| `descriptors.csv` | one row per SNUMAT record |
| `pairs.csv` | one row per same-composition polymorph pair |
| `tables/` | model performance, descriptor effects, hypothesis verdicts, case checks |

## 1. Structures

Each SNUMAT record carries `atoms = {lattice_mat, coords (fractional), elements}`, which is
turned straight into `Structure(lattice_mat, elements, coords, coords_are_cartesian=False)`.
No symmetrisation, no primitive reduction: the cell as relaxed by SNUMAT is used.

## 2. Descriptor definitions

**Cation / anion assignment.** Of the distinct elements in the record, the anions are the most
electronegative element plus every element within 0.4 Pauling units of it (`Element.X`); the
remaining elements are cations. The rule is compositional, so the two members of a polymorph pair
always get the same assignment, and every cation–anion quantity below is comparable inside a pair.
Elemental solids have no cation and the cation-based entries are empty for them.

**Neighbour list.** One `CrystalNN()` call per structure (`get_all_nn_info`), with its default
settings and no oxidation-state decoration (so Van der Waals/atomic radii are used). Every
coordination, bond, connectivity and dimensionality quantity below is derived from that one
neighbour list, so they are mutually consistent.

*Symmetry* — `SpacegroupAnalyzer(symprec=0.1, angle_tolerance=5)`: `sg_number`, `sg_symbol`,
`crystal_system`, `lattice_type`, `point_group`, `n_symops` (number of symmetry operations),
`bravais` (crystal system plus the centring letter of the Hermann–Mauguin symbol) and
`centrosymmetric` (point group is one of the 11 Laue classes). The record's own `Space_group_rlx`
is kept as well and is what defines a polymorph cell; `sg_number` is the independent spglib value.

*Coordination* — `cn_cat_{mean,min,max,std,spread}` and `cn_an_*`: the CrystalNN coordination
number over cation and over anion sites (`spread` = max − min); `cn_all_*` over all sites.
`dom_cn_cat` is the modal cation coordination number, `dom_poly_cat` its nominal polyhedron
(4 tetrahedral, 6 octahedral, …; a pure coordination-number label, no geometry test), and
`dom_poly_frac` the fraction of cation sites that carry it.

*Connectivity* — a cation-centred polyhedron is the cation plus its anion neighbours, each
neighbour identified by (site index, periodic image). Two polyhedra, including a polyhedron and a
lattice translation of itself, are connected if they share at least one anion vertex: 1 shared
vertex = corner-sharing, 2 = edge-sharing, ≥3 = face-sharing. Connections are enumerated over all
cation pairs and all lattice translations that give a shared vertex, each counted once.
`frac_corner`, `frac_edge`, `frac_face` are the fractions of connections of each type,
`mean_shared_vertices` and `max_shared_vertices` the mean and maximum number of shared vertices
per connection, `conn_per_cation` the number of connections per cation site, and
`frac_isolated_poly` the fraction of cation polyhedra that share no vertex with any other
polyhedron (isolated units).

*Dimensionality* — `get_dimensionality_larsen` (pymatgen.analysis.dimensionality) on a
`StructureGraph` built from the same neighbour list: `dim_all` on all bonds, `dim_cat_an` on
cation–anion bonds only (the bonded network the question is about). 0/1/2/3.

*Bonding geometry* — over cation–anion bonds only: `bond_cat_an_mean`, `bond_cat_an_std`,
`bond_cat_an_rel_std` (std/mean), and the Baur distortion index per cation polyhedron,
mean |d_i − d̄| / d̄, averaged over cations (`distortion_index`) and its maximum
(`distortion_index_max`). Bond valence sums use `pymatgen.analysis.bond_valence.calculate_bv_sum`
with oxidation states guessed from the composition (`Composition.oxi_state_guesses`), summed over
the anion neighbours of each cation: `bvs_cat_mean` and `bvs_dev_mean` = mean |BVS − formal
valence|. `oxi_ok` flags the records where the guess succeeded.

*Chemistry* — fractional-composition weighted: `en_cat_mean`, `en_an_mean`, `en_diff`
(anion − cation mean electronegativity), `ionicity` = 1 − exp(−0.25 Δχ²) (Pauling),
`en_std_all`, `anion_frac`, `nval_per_atom` (valence electrons per atom: all electrons of the
highest principal shell plus an incomplete (n−1)d or (n−2)f sub-shell), `radius_ratio` (mean
cation over mean anion atomic radius) and `mean_atomic_radius`.

*Packing* — `volume`, `vol_per_atom`, `density`, and `packing_fraction` = Σ (4/3)πr³ / V with r the
`Element.atomic_radius` (calculated radius as fallback).

**These chemistry descriptors are properties of the composition alone**, so inside a
same-composition pair their difference is exactly zero and they cannot explain anything about the
polymorph gap difference. They are kept in `descriptors.csv` because stage 2 needs them, and they
enter the pair analysis only as context (the `c_*` columns of `pairs.csv`).

## 3. Cells and pairs

A *cell* is one (canonical composition key, `Space_group_rlx`) combination. The composition key is
the published one, `'|'.join(f'{el}:{frac:.12f}')` over the fractional composition. Where a cell
holds several records (different ICSD entries) the cell gap is the **median** over them, as in
`benchmark/README.md`, and the descriptors are those of the single record whose HSE gap is closest
to that median — so the descriptors always describe one real structure.

A *pair* is two cells of the same composition with different relaxed space groups. The pair is
oriented by space-group number: member *a* has the lower number, member *b* the higher, and every
difference is *b − a*. Each composition carries equal total weight, split evenly over its pairs
(`weight` in `pairs.csv`), exactly as the S5 experimental pair analysis does; the weighted numbers
are the ones quoted.

## 4. Models

The target is odd under swapping the two members, so every model is fitted on both orientations of
each pair (features and target negated) and with no intercept. Ridge (α=10 on the full descriptor
set, α=1 on the six-descriptor model) is fitted on standardised differences, so its coefficients
read as eV per standard deviation of that descriptor difference. The gradient-boosted model is
`HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=4)`, which takes the
missing values directly; for ridge a missing difference is set to 0 and flagged by an indicator
column. Cross-validation is 5-fold `GroupKFold` with the group being, in turn, the composition
key, the element system (the set of elements — a whole chemistry held out) and the anion system.
Permutation importance is computed out of fold on the held-out composition folds, in eV of MAE.
Univariate slopes are weighted no-intercept fits (the odd-function form y = s·x) with 95%
confidence intervals from 400 bootstrap resamples **of compositions**, not of pairs.

## 5. Counts

* **10,481 of 10,481** SNUMAT records rebuilt into a pymatgen `Structure` and described.
  **No failures** (`descriptor_failures.csv` is not written). 392 records (3.7%) have no cation
  under the electronegativity rule — elemental solids and a few all-nonmetal compositions — and
  their cation-based entries are empty; 88.2% of records got an oxidation-state guess and
  therefore a bond valence sum.
* 10,186 cells (composition × relaxed space group) over 8,730 compositions.
* **929 compositions carry more than one relaxed space group, giving 2,838 pairs**
  (median 1 pair per composition, mean 3.05, maximum 325 for one composition — hence the
  equal-weight-per-composition weighting). 842 distinct element systems, 71 anion systems.
  2,612 of the 2,838 pairs have a cation in both members.
* spglib at symprec 0.1 reproduces SNUMAT's `Space_group_rlx` for 89.0% of records.
* Chemistry descriptor differences inside a pair: max 1.8e-15 — zero, as they must be.
* Scale of the target: mean |ΔE_g(HSE)| = 0.541 eV, median 0.272, sd 0.933. HSE optical
  mean |Δ| 1.118 eV, GGA 0.477 eV.

## 6. Which descriptor differences explain the gap difference

Held-out performance for ΔE_g(HSE) (`tables/model_performance.csv`, composition-weighted):

| model | held out | R² | MAE (eV) | MAE of predicting 0 | direction agreement |
|---|---|---|---|---|---|
| ridge, 35 descriptors | composition | 0.058 | 0.407 | 0.424 | 0.565 |
| ridge, 35 descriptors | element system | 0.061 | 0.407 | 0.424 | 0.579 |
| ridge, 35 descriptors | anion system | 0.049 | 0.411 | 0.424 | 0.551 |
| gradient boosting | composition | 0.046 | 0.416 | 0.424 | 0.599 |
| gradient boosting | element system | 0.112 | 0.409 | 0.424 | 0.593 |
| gradient boosting | anion system | 0.007 | 0.430 | 0.424 | 0.594 |
| ridge, six descriptors | composition | 0.057 | 0.407 | 0.424 | 0.592 |
| ridge, six descriptors | element system | 0.060 | 0.407 | 0.424 | 0.602 |

For the GGA difference the picture is the same (R² 0.046–0.085). For the **optical** difference
the descriptors carry almost nothing (ridge R² ≤ 0.010; the optical gap difference is dominated by
the direct/indirect character, not by the geometry).

**This is the central number: structural descriptor differences explain about 6% of the variance
of the polymorph gap difference, and they lower the mean absolute error from 0.424 eV (predict
zero) to 0.407 eV — 4%.** Direction is called right 56–60% of the time against 50% for a coin.
Holding out whole chemistries costs nothing, because there is almost nothing to lose: the little
signal that exists is a general, chemistry-independent trend, not a transferable fine structure.

Effect sizes (`tables/descriptor_effects.csv`; slope from a weighted odd-function fit with a
composition-clustered 95% CI; `r2_alone` is that single descriptor's weighted R²):

| descriptor difference | slope | 95% CI | Spearman | R² alone |
|---|---|---|---|---|
| mean cation–anion bond length | −2.09 eV/Å | [−2.81, −1.36] | −0.282 | 0.060 |
| cation bond valence sum | +0.93 eV per v.u. | [0.60, 1.36] | +0.234 | 0.053 |
| mean cation coordination number | −0.166 eV per unit | [−0.236, −0.098] | −0.294 | 0.048 |
| mean coordination number, all sites | −0.200 eV per unit | [−0.275, −0.124] | −0.221 | 0.044 |
| mean shared vertices per connection | −0.529 eV per unit | [−0.765, −0.345] | −0.244 | 0.040 |
| number of symmetry operations | −0.0092 eV per operation | [−0.0126, −0.0065] | −0.254 | 0.038 |
| face-sharing fraction | −1.33 eV per unit | [−2.24, −0.55] | −0.141 | 0.025 |
| corner-sharing fraction | +0.39 eV per unit | [0.09, 0.68] | +0.213 | 0.004 |
| bonded-network dimensionality | +0.026 eV per step | [−0.090, +0.169] | +0.047 | −0.010 |
| polyhedral distortion index | — (see H5) | | −0.081 | — |

Permutation importance (out of fold, eV of MAE) puts the same four first — bond valence sum,
number of symmetry operations, mean bond length, mean shared vertices — but every single value is
below 0.003 eV, i.e. no descriptor is worth 3 meV of MAE on its own.

**These descriptors are not independent.** At fixed composition, corr(Δ mean bond length,
Δ mean cation CN) = **+0.866**: a polymorph with higher coordination has longer bonds, necessarily.
corr(Δ packing fraction, Δ volume per atom) = −0.82, corr(Δ face-sharing, Δ CN) = +0.56.
So there is **one** structural axis in the data — how dense the cation coordination shell is — and
the apparent effects of bond length, coordination number, sharing and symmetry are that one axis
seen from different directions. A joint fit gives
ΔE_g = −1.19·Δd(cation–anion) − 0.103·ΔCN, and the single-variable slopes collapse when the other
is held fixed (bond-length slope +0.31 eV/Å at |ΔCN| < 0.1, n = 1461; CN slope −0.091 eV at
|Δd| < 0.02 Å, n = 1299). Only the pair of them together carries the 6%.

## 7. Hypothesis by hypothesis

All contrasts are composition-weighted with composition-clustered 95% CIs
(`tables/hypotheses.csv`). Each pair is oriented so that the sign asked about is positive if the
hypothesis holds.

1. **Lower bonded-network dimensionality gives a larger gap — NOT SUPPORTED.**
   Only 359 of 2,838 pairs (12.6%) change the cation–anion network dimensionality at all; 87.4%
   of same-composition polymorph pairs have the same dimensionality. Over those 359 pairs the
   lower-dimensional member's gap is **−0.061 eV** [−0.262, +0.139] relative to the
   higher-dimensional one — zero within error, and only 46.9% of them are positive. For a drop of
   one dimension (n = 272) it is −0.113 eV [−0.391, +0.137]; for a drop of two (n = 57) +0.256 eV
   [−0.033, +0.700], the only sub-case that leans the hypothesis's way, and its median is negative.
   The FAPbI3 α→δ case is real but it is not the rule: in SNUMAT the dimensionality of the bonded
   network rarely changes between polymorphs, and when it does the gap does not follow it.
2. **More face- or edge-sharing gives a larger gap than corner-sharing — SUPPORTED WITH THE
   OPPOSITE SIGN.** The member with more face+edge sharing has the **smaller** gap:
   −0.191 eV [−0.309, −0.075] for |Δ(face+edge fraction)| > 0.1 (n = 949), −0.410 eV
   [−0.658, −0.177] for |Δ| > 0.5 (n = 187), and −0.525 eV [−0.895, −0.170] in the
   corner-dominated contrast (n = 134). The effect is real and significant and it runs against the
   hypothesis. Physically it is the coordination axis again: face- and edge-sharing accompanies
   higher coordination, which in SNUMAT means a smaller gap.
3. **Shorter cation–anion bonds give a smaller gap — SUPPORTED WITH THE OPPOSITE SIGN, AND
   CONFOUNDED.** The shorter-bonded member has the **larger** gap: +0.329 eV [0.233, 0.435] for
   |Δd| > 0.05 Å (n = 956) and +0.542 eV [0.273, 0.812] for |Δd| > 0.2 Å (n = 212). But at fixed
   composition a shorter mean bond means a lower coordination number (r = +0.866), so this is not
   a statement about orbital overlap at fixed coordination: restricted to pairs with
   |ΔCN| < 0.1 the slope is +0.31 eV/Å with r = −0.12, i.e. nothing. The bandwidth argument is not
   wrong, it is simply not what varies between polymorphs of one composition.
4. **Higher cation coordination gives a smaller gap — SUPPORTED.** The higher-coordinated member's
   gap is **−0.282 eV** [−0.370, −0.202] for |ΔCN| > 0.25 (n = 1105) and −0.377 eV
   [−0.487, −0.264] for |ΔCN| ≥ 1 (n = 870); 64–69% of pairs follow it. This is the one hypothesis
   that survives as stated, and it is the same axis as 2 and 3.
5. **Larger polyhedral distortion gives a larger gap — NOT SUPPORTED.** +0.056 eV
   [−0.052, +0.168] for |Δ(distortion index)| > 0.01 (n = 931) and −0.315 eV [−0.722, +0.051] for
   |Δ| > 0.05 (n = 104). Zero within error, with the sign reversing for the strongly distorted
   subset.

A floor worth stating: for the 1,147 pairs where nothing local changes much
(|ΔCN| < 0.1, |Δd| < 0.02 Å, same dimensionality) the mean |ΔE_g| is still **0.227 eV**
(median 0.100), and for the 2,479 pairs at equal dimensionality it is 0.496 eV. Two polymorphs
can therefore differ by a few tenths of an eV with no signature in any of these descriptors.

## 8. Ga2O3 and FAPbI3

`tables/case_checks.csv`. For each case the model is refitted with the whole chemistry removed
from the training pairs (every pair containing Ga for Ga2O3 — 2,807 pairs left; every pair
containing Pb or I for FAPbI3 — 2,512 left), so these are transfer predictions.

**Ga2O3.** SNUMAT holds exactly two polymorphs, β (C2/m, #12) and α (R-3c, #167), with HSE gaps
3.538 and 4.012 eV. The computed difference, α − β = **+0.474 eV**, has the sign and roughly the
size of the measurement (α 4.86–5.32 eV versus β 4.36–5.00 eV in the provenance set; +0.3 to
+0.5 eV). The descriptors change exactly as expected — α has CN 6 against β's 5, longer bonds
(2.031 vs 1.975 Å), some face-sharing (0.077 against 0) and more shared vertices. **The fitted
relation then predicts −0.12 eV (six-descriptor) or −0.09 eV (full set): the wrong sign.** The
experimental reference CIFs (α R-3c, β C2/m) give the same answer, −0.11 / −0.12 eV. κ-Ga2O3 has
no reference structure in the experimental set and is not in SNUMAT, so the three-way comparison
cannot be made.

**FAPbI3.** Not in SNUMAT (organic cation), so the two phases come from the experimental
reference CIFs — α P-43m (#215) and δ P6₃mc (#186) — reduced to their **PbI3 framework**: the
disordered formamidinium cation is dropped and, in α, the 4-fold split iodine site
(occupancy 0.25, 0.24 Å off the ideal position) is averaged back to the ideal octahedral site.
The descriptors come out textbook: α is a 3D network of **corner-sharing** PbI6 octahedra
(dimensionality 3, 1.00 shared vertex per connection, zero distortion, d = 3.185 Å), δ a 1D
network of **face-sharing** octahedra (dimensionality 1, 3.00 shared vertices, d = 3.212 Å), at
essentially the same volume per framework atom (64.6 vs 64.2 Å³). The measured difference is
α − δ = 1.47 − 2.43 = **−0.96 eV**. **The fitted relation predicts +0.47 eV (six-descriptor) or
+0.17 eV (full set): again the wrong sign.** It must, given section 7: the SNUMAT average says
more face-sharing means a smaller gap, and FAPbI3 is the opposite.

**So the relation fitted on SNUMAT reproduces neither manuscript case.** The two cases are not
noise — they are each about 0.5–1 eV, and the DFT numbers for Ga2O3 get the sign right. What fails
is the attempt to read those differences off coordination, connectivity and dimensionality
descriptors.

## 9. What the data does and does not support

* It supports **one** structural statement at fixed composition: polymorphs with a denser cation
  coordination shell — higher coordination number, more shared polyhedral vertices, longer
  individual bonds — have **smaller** gaps, about **−0.3 eV per unit of mean cation coordination
  number**, with coordination number, bond length and sharing fraction being three views of that
  single axis.
* It does **not** support the dimensionality hypothesis (the dimensionality of the bonded network
  barely varies between polymorphs of one composition, and where it does the gap does not follow),
  it does not support the distortion hypothesis, and it refutes, with the opposite sign, both the
  "face/edge-sharing widens the gap" and the "shorter bonds narrow the gap" hypotheses as stated.
* The total explanatory power is small and does not grow when a gradient-boosted model is allowed
  to find interactions: **R² ≈ 0.06, 4% off the MAE of predicting no difference at all, direction
  right 56–60% of the time.** It is the same with whole chemistries held out, so what little there
  is would transfer — there is just very little.
* Two polymorphs of one composition differ by 0.54 eV on average, and 0.23 eV even when every
  descriptor here is nearly identical. Most of the polymorph gap difference lives in band-structure
  detail (which orbitals sit at the edges, direct versus indirect character, octahedral tilting
  patterns, long-range electrostatics) that coordination-and-connectivity descriptors do not see.
* For stage 2 this is the honest expectation: replacing the raw space-group vector with these
  descriptors removes an overfitting input and gives the structure pathway a physically meaningful,
  always-computable one, but it should not be expected to deliver large pair-MAE gains. On this
  evidence the structure pathway can at best move a polymorph prediction by a few tenths of an eV
  in the right direction a little over half the time.
