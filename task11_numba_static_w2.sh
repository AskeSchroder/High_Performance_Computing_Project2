# task7a_numba_static_w2.sh
#!/bin/bash
#BSUB -J t7a_w2
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task7a_numba_static_w2_%J.out
#BSUB -eo task7a_numba_static_w2_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
python run_subset_7a.py 100 --workers 2 --time
