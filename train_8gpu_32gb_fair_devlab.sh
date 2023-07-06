#!/bin/bash

#SBATCH --partition=devlab
#SBATCH --job-name=robotics
#SBATCH --output=slurm_logs/robotics-%j.out
#SBATCH --error=slurm_logs/robotics-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=512gb
#SBATCH --constraint=volta32gb

python -m torch.distributed.launch --nproc_per_node 8 --master_port $RANDOM main_trajectory.py "$@"