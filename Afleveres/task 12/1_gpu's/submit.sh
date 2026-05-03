#!/bin/bash
#BSUB -J full_run_gpuv100
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -R "rusage[mem=10GB]"
##BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -gpu "num=1:"
#BSUB -R "span[hosts=1]"
#BSUB -o run%J.out
#BSUB -e run%J.err
##BSUB -u s211930@student.dtu.dk
##BSUB -B
##BSUB -N 


prefix="12_full_dataset_non_exclusive_gpuv" 
current_dir="/zhome/8c/6/163231/python/task_12/gpuv"
py_path="run_full_12.py" 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ${current_dir} || exit 1

python ${py_path} 4571 \
	--tpb-x 16 \
	--tpb-y 16 \
	--large-block 2500 \
	--medium-block 1250 \
	--small-block 625 \
	--near-block 312 \
	--time > ${prefix}.csv
