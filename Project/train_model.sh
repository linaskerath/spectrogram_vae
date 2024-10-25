#!/bin/bash

#SBATCH --partition=titans          # Specify the partition
#SBATCH --gres=gpu:Ampere:2         # Request 2 Ampere GPUs
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Request 1 task (usually 1 per node)
#SBATCH --cpus-per-task=8           # Number of CPUs per task
#SBATCH --mem=64G                   # Memory per node
#SBATCH --job-name=my_gpu_job       # Job name


## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.9
python main.py

echo "Done: $(date +%F-%R:%S)"