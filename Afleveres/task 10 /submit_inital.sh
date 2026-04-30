#!/bin/bash
#BSUB -J cupy_nsys
#BSUB -q c02613
#BSUB -W 00:10
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err

prefix="cupy_nsys" 
current_dir="python/task_10/" 
py_path="run_subset_10.py"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
module swap cuda/13.0.2

cd /zhome/8c/6/163231/${current_dir} || exit 1

nsys profile -o ${prefix}_profile python ${py_path} 100 --max-iter 5000 --time > ${prefix}.csv

nsys stats ${prefix}_profile.nsys-rep > ${prefix}_stats.txt
