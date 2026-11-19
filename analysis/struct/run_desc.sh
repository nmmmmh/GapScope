#!/bin/bash
# Launch the descriptor computation over NSH shards.  One BLAS thread per process:
# the default OpenBLAS 128 threads per process exhausts RLIMIT_NPROC (4096) on this
# shared machine and makes new ssh sessions fail.
cd /data/czx/GapScope_Revision_20261113/struct || exit 1
PY=/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python
NSH=${1:-24}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 OPENBLAS_MAIN_FREE=1
mkdir -p shards
rm -f shards/d*.csv shards/log*.txt
for s in $(seq 0 $((NSH - 1))); do
  nohup $PY -I desc_worker.py "$s" "$NSH" "shards/d$s.csv" > "shards/log$s.txt" 2>&1 &
done
echo "launched $NSH shards"
