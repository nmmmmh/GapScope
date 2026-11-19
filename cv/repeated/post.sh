#!/bin/bash
# post.sh CODE_DIR GPU SEED... : when all 15 Roost predictions of (seed, holdout) exist, run roost_cal_pred.py (ROOST_CAL inputs) once
C=$1; G=$2; shift 2; cd $C; source ./env.sh
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
P0=$(dirname $C); [ -d $P0/s$1 ] || P0=$(dirname $C)
while true; do left=0
  for s in "$@"; do P=$P0/s$s; for v in comp chem; do
    [ -f $P/ft/cal/done_$v ] && continue; left=1
    [ $(ls $P/roost_runs/pred_${v}_f*_s*.csv 2>/dev/null | wc -l) -eq 15 ] || continue
    $PY -B $C/roost_cal_pred.py $P $P/ft/cal $v 1,2,3,4,5 > $P/logs/calpred_$v.log 2>&1 && touch $P/ft/cal/done_$v
  done; done
  [ $left = 0 ] && break; sleep 60
done
