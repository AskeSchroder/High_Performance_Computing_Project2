# task7_numba_only.sh
#!/bin/bash
#BSUB -J t7_numba_only
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task7_numba_only_%J.out
#BSUB -eo task7_numba_only_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate

python run_subset_7.py 20 --time
