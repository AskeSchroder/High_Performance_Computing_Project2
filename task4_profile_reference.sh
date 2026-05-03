# task4_profile_reference.sh
#!/bin/bash
#BSUB -J t4_prof
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:45
#BSUB -oo task4_profile_reference_%J.out
#BSUB -eo task4_profile_reference_%J.err

cd ~/wall_heating
source ~/venv_hpc/bin/activate
mkdir -p results

python -m kernprof -l -v run_subset_4.py 1 > results/task4_profile_${LSB_JOBID}.txt 2>&1
