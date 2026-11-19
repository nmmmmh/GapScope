# Repeated fold assignments (cv_rep/, 2026-11-16)

Purpose: the grouped 5-fold CV of cv/ repeated on 4 new fold assignments (seeds 20261114-20261117), so that the distributional model
and its baselines can be reported as mean ± SD over 5 partitions (original 20261113 + 4 new). Nothing in cv/ was modified.

## What was run
- Folds: code/make_folds.py = the 'groups, folds' section of cv/build_data.py copied verbatim, SEED from argv; tier-A/tier-B tables are
  read from cv/data (symlinked into s<seed>/data, not rebuilt). Seed 20261113 reproduces folds_comp.csv, folds_chem.csv,
  data/inner_val_comp.json and data/inner_val_chem.json byte for byte, and cv/roost_in/*.csv (20 files) byte for byte.
  Inner validation seed = SEED+100+fold as in build_data.py.
- Per partition s<seed>/: roost_prep.py -> roost_in/, prep_ft.py -> ft/ft_data.pt (copies of cv/ scripts, CV path from env CVDIR).
- GS_SELCAL_PRE_NZ ("this work"): GapScope_cv_remote/ft/gsft4.py (comp) and gsft7.py (chem), copied unchanged, flags as in the published
  info json: --init mp --val group20 --crps 1.0 --l1 0.5 --sel selcal --epochs 300 --patience 40 --nodiag --hse_arm NZ, seeds 17/29/41,
  own NZ HSE pretraining per (holdout, fold, seed). group20 validation keeps the recipe constant default_rng(20261113 + 7*fold).
  All 120 runs on c136-152 (roost152 env, as published), 4 GPUs shared with other running jobs.
- ROOST: release xfer-mp fine-tune, 250 epochs, seeds 17/29/41 (run_roost_cv.py on c137-13 / run_roost_cv_r.py on remotes, unchanged except
  CV path). ROOST_CAL inputs: roost_cal_pred.py (predtrain_*, release internal 20% split). s20261114 on c137-13; s20261117 on c136-152;
  s20261115/16: 16 runs on c137-105, 44 on c136-152 (105 disk I/O stalled; its 16 unfinished runs were killed and rerun from scratch on 152).
- EASYUQ_KS: easyuq.py on each partition's Roost predictions.
- Evaluation: code/eval_part.sh = easyuq.py + run_eval_v4.py (evaluate_cv_v4 unchanged) --new GS_SELCAL_PRE_NZ --scale --easyuq
  --only GS_SELCAL_PRE_NZ_SC,ROOST_CAL,EASYUQ_KS (--only only drops other sets from scoring; the SC width factor is fitted before it).
  Original partition (s20261113): archived cv/ predictions (roost_runs, ft/cal, gs_out_v2 symlinked) re-evaluated with the copied code
  without --only: all 8 (holdout, model) pooled CRPS/cov80/width80 equal the published values, max per-unit CRPS difference 0.0
  (check_original_partition.csv).
- Summary: code/summarize.py, aggregation as cv/paired_bootstrap_final.py (tier A per composition, hierarchical weights normalised within
  composition; pooled = 85 tier-A compositions + 2,467 tier-B units, equal weight); paired composition bootstrap 4000 draws, seed 20261113.

## Outputs
summary_partitions.csv (per partition/holdout/model: pooled CRPS, cov80, width80, pooled MAE, tier-A MAE, tier-A CRPS),
summary_paired_partitions.csv (this work - each reference per partition, with 95% paired bootstrap intervals),
summary_mean_sd.csv (mean and SD, ddof=1, over the 5 partitions; rows this_work_minus_* are the paired differences),
check_original_partition.csv, s<seed>/predictions_summary_rep_{comp,chem}.csv (per-unit scores), s<seed>/logs/.

## Wall time
About 2 h 10 min on 2026-11-16 (01:10-03:20 server time): GS runs 156-438 s each (median 231 s); Roost runs median ~13-14 min with 4-8
concurrent per GPU; evaluation ~40 min per partition (two holdouts in parallel).
