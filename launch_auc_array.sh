#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%A_%a.txt # file to save job's STDOUT & STDERR
#SBATCH --export=NONE   # Purge the job-submitting shell environment


# e.g. parallel -k echo {} ::: $(seq 1 50) ::: RF XGB ::: all CN_CP NN_CP ::: none smote adasyn > manifest_general
# e.g. sbatch -a 1-150 -c 16 lanuch.sh manifest

manifest="${1:?ERROR -- must pass a manifest file}"
taskid=$SLURM_ARRAY_TASK_ID
trial_idx=$(getline $taskid $manifest | cut -f 1 -d ' ')
model=$(getline $taskid $manifest | cut -f 2 -d ' ')
file=$(getline $taskid $manifest | cut -f 3 -d ' ')
layer=$(getline $taskid $manifest | cut -f 4 -d ' ')
maxiter=$(getline $taskid $manifest | cut -f 5 -d ' ')


module purge
# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate tree

python auc.py $trial_idx $model $file $layer $maxiter
