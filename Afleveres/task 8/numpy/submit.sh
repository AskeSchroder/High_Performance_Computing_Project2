#!/bin/bash
#BSUB -J 8_numpy
#BSUB -q c02613
#BSUB -W 00:10
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err

prefix="numpy" 
current_dir="python/task_8/numpy" 
py_path="run_subset_8.py"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd /zhome/8c/6/163231/${current_dir} || exit 1

python ${py_path} 100 --backend numpy --max-iter 5000 --time > ${prefix}.csv
