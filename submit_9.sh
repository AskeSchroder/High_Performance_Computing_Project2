#!/bin/bash 

#job name 
#BSUB -J cupy_run

# Queue name 
#BSUB -q c02613

# Wall-clock time 
#BSUB -W 00:30 

# Ressources (memmory) 
#BSUB -R "rusage[mem=10GB]"

# Number of cpu cores 
#BSUB -n 4

# Number of gpu cores 
#BSUB -gpu "num=1:mode=exclusive_process"

#Number of hosts
#BSUB -R "span[hosts=1]"

# Output file (stdout) 
#BSUB -o cupy_run%J.out

# Output file (stderr)
#BSUB -e cupy_run%J.err

module load cuda/11.6
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd /zhome/8c/6/163231/courses/python_high_performance || exit 1

python run_subset_9b.py 20 --max-iter 5000 --time > cupy_20.csv

