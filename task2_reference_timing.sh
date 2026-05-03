#!/bin/bash
#BSUB -J t2_ref
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -oo task2_reference_timing_%J.out
#BSUB -eo task2_reference_timing_%J.err

python run_subset_2.py 20 --time
