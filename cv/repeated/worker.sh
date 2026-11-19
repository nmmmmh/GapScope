#!/bin/bash
# copy of GapScope_cv_remote/ft/worker.sh (per-GPU queue): worker.sh <queue> <gpu> <max concurrent>; env from ./env.sh
cd "$(dirname "$0")"
Q=$1; N=$3; i=0
source ./env.sh
export CUDA_VISIBLE_DEVICES=$2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
touch "$Q"
while true; do
  total=$(wc -l < "$Q")
  if [ $i -lt $total ]; then
    while [ $(jobs -rp | wc -l) -ge $N ]; do sleep 10; done
    i=$((i+1)); cmd=$(sed -n "${i}p" "$Q")
    [ -n "$cmd" ] && { bash -c "$cmd" & }
    echo "$i" > "$Q.pos"
  else
    [ -f "$Q.stop" ] && break
    sleep 20
  fi
done
wait
