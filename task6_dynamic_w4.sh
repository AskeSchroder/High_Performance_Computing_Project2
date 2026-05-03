#!/bin/bash
#BSUB -J t6d_w4
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task6_dynamic_w4_%J.out
#BSUB -eo task6_dynamic_w4_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
python run_subset_6.py 100 --workers 4 --time
