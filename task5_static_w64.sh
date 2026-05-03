#!/bin/bash
#BSUB -J t5s_w64
#BSUB -q hpc
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task5_static_w64_%J.out
#BSUB -eo task5_static_w64_%J.err

python run_subset_5.py 100 --workers 64 --time
