#!/bin/bash
#BSUB -J 11_improve_10000
#BSUB -q c02613
#BSUB -W 00:30
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err

prefix="subset_11"
current_dir="/zhome/8c/6/163231/python/task_11/b10000" 
py_path="run_subset_11.py" 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ${current_dir} || exit 1

python ${py_path} 100 \
	--tpb-x 16 \
	--tpb-y 16 \
	--large-block 10000 \
	--medium-block 5000 \
	--small-block 2500 \
	--near-block 1250 \
	--time > ${prefix}.csv


