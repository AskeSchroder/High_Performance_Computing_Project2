# task7a_numba_static_w8.sh
#!/bin/bash
#BSUB -J t7a_w8
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task7a_numba_static_w8_%J.out
#BSUB -eo task7a_numba_static_w8_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
python run_subset_7a.py 100 --workers 8 --time
