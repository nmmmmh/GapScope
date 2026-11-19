Q=/data/czx/GapScope_Revision_20261113/cv_rep; C=$Q/code; cd $C
cp $Q/../cv_rep/code/../make_folds.py . 2>/dev/null; true
echo 'export PY=/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python GSFT_ROOT=/data/czx/GapScope_Revision_20261113/roost' > env.sh
s=20261114; : > q_r0.txt; mkdir -p $Q/s$s/logs
for v in comp chem; do for f in 1 2 3 4 5; do for sd in 17 29 41; do
  echo "CVDIR=$Q/s$s \$PY -B $C/run_roost_cv.py $v $f $sd > $Q/s$s/logs/roost_${v}_${f}_${sd}.log 2>&1" >> q_r0.txt; done; done; done
nohup bash worker.sh $C/q_r0.txt 0 5 > /dev/null 2>&1 &
sleep 2; wc -l q_r0.txt
