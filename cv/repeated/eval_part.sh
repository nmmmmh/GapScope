#!/bin/bash
# eval_part.sh PARTDIR VARIANT : EasyUQ on the partition's Roost predictions, then evaluate_cv_v4 (run_eval_v4) for ROOST, ROOST_CAL, EASYUQ_KS, GS_SELCAL_PRE_NZ_SC
export CVDIR=$1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EQPAT=easyuq_$2.pkl
PY=/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python
C=/data/czx/GapScope_Revision_20261113/cv_rep/code
cd $1
$PY -B $C/easyuq.py $2 > $1/logs/easyuq_$2.log 2>&1
$PY -B $C/run_eval_v4.py --variants $2 --new GS_SELCAL_PRE_NZ --scale --easyuq --only GS_SELCAL_PRE_NZ_SC,ROOST_CAL,EASYUQ_KS --suffix _rep_$2 > $1/logs/eval_$2.log 2>&1
echo finished $2
