# Error floor on the compilation (2026-11-15)
Python: BandgapFusion_RoostReproductionWork venv, -I, OMP_NUM_THREADS=1. CPU only, seconds.
Input: cv/data/tierB_all_rows.csv (3,895 rows); identical (composition, value) collapsed -> 3,728 rows, 2,467 compositions, 452 with >=2 values.
Polymorph count = distinct space groups over both SNUMAT exports (benchmark/snumat_tidy.csv + hse_csv_keyed.csv).
- c1_floor.py: floor per duplicated composition 0.083 eV (paper, 85 traced compositions: 0.080). 0.1 eV rule removes 318 compositions and 95.2% of the floor (paper: 38, 94%).
  Multi (>=2 SG) vs single: 0.088 vs 0.083 eV, MWU P=0.33, ratio 1.06 [0.89,1.26]; no trend with SG count.
- c2_proxy.py: on the traced set the same SNUMAT proxy separates 0.146 vs 0.050 eV (labels: 0.143 vs 0.029), so the proxy works; the null is in the compilation.
- c3_overlap.py: 21 compositions in both sets; compilation values carry non-phase scatter of the same size (e.g. MoO3 3.54-3.75 vs traced 2.45-2.94; Ga2O3 compilation lacks alpha).
