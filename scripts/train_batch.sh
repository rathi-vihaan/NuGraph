#!/bin/bash
#SBATCH -J nugraph_train
#SBATCH -t 12:00:00
<<<<<<< HEAD
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
=======
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH -q normal
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G
>>>>>>> 1c31f447cde7c81339dec81a3ba2f1d03fefab85

source /etc/profile.d/conda.sh
conda activate /net/projects2/fermi2526/conda/nugraph-25-10
which python
<<<<<<< HEAD
srun python scripts/train.py $@
=======
ulimit -n 65536
echo "fd limit set to: $(ulimit -n)"
srun python scripts/train.py $@
>>>>>>> 1c31f447cde7c81339dec81a3ba2f1d03fefab85
