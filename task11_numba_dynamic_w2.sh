# task7b_numba_dynamic_w2.sh
#!/bin/bash
#BSUB -J t7b_w2
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task7b_numba_dynamic_w2_%J.out
#BSUB -eo task7b_numba_dynamic_w2_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
python run_subset_7b.py 100 --workers 2 --time
