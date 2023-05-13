#!/bin/bash

#SBATCH --partition=learnfair
#SBATCH --job-name=robotics
#SBATCH --output=slurm_logs/robotics-%j.out
#SBATCH --error=slurm_logs/robotics-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128gb
#SBATCH --constraint=volta32gb

python train.py "$@"