#!/bin/bash
#SBATCH -J nugraph_train
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH -p general
#SBATCH -N 1                   
#SBATCH --ntasks-per-node=8      
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -q normal
#SBATCH --cpus-per-task=4       
#SBATCH --mem=256G

source /etc/profile.d/conda.sh
conda activate /net/projects2/fermi2526/conda/nugraph-25-10
which python
srun python scripts/train.py $@