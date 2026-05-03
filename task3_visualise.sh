#!/bin/bash
#BSUB -J t3_vis
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -oo task3_visualize_%J.out
#BSUB -eo task3_visualize_%J.err

python visualise_results.py 3 --outdir figures_task3
