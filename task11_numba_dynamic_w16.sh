# task7b_numba_dynamic_w16.sh
#!/bin/bash
#BSUB -J t7b_w16
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task7b_numba_dynamic_w16_%J.out
#BSUB -eo task7b_numba_dynamic_w16_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
python run_subset_7b.py 100 --workers 16 --time
