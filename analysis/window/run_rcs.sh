for s in 20261113 20261114 20261115 20261116 20261117; do
  OMP_NUM_THREADS=1 /home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python window_rep_rcs.py /data/czx/GapScope_Revision_20261113/cv_rep/s$s > rcs_$s.log 2>&1
done
echo ALLDONE > rcs_done.txt
