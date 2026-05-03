#!/bin/bash
#BSUB -J t5s_w2
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task5_static_w2_%J.out
#BSUB -eo task5_static_w2_%J.err

python run_subset_5.py 100 --workers 2 --time
