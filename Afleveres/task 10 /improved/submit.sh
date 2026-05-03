#!/bin/bash
#BSUB -J cupy_improved
#BSUB -q c02613
#BSUB -W 00:30
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err

prefix="cupy_improved" 
current_dir="/zhome/8c/6/163231/python/task_10/improved" 
py_path="run_subset_9_improved.py"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ${current_dir} || exit 1

python ${py_path} 100 --time > ${prefix}.csv
