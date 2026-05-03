#!/bin/bash
#BSUB -J t6d_w64
#BSUB -q hpc
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task6_dynamic_w64_%J.out
#BSUB -eo task6_dynamic_w64_%J.err

python run_subset_6.py 100 --workers 64 --time
