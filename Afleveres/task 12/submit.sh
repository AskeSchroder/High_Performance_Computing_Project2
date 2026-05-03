#!/bin/bash
#BSUB -J 12_p2
#BSUB -q gpuv100
#BSUB -W 01:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 8
#BSUB -gpu "num=2"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err

prefix="12_p2" 
current_dir="/zhome/8c/6/163231/python/task_12/parallel/p2"
py_path="run_subset_12_p.py" 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ${current_dir} || exit 1

python ${py_path} 4571 \
	--backend cuda \
	--num-gpus 2 \
	--tpb-x 16 \
	--tpb-y 16 \
	--large-block 2500 \
	--medium-block 1250 \
	--small-block 625 \
	--near-block 312 \
	--time > ${prefix}.csv
