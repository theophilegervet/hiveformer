#!/bin/sh

#main_dir=peract
#use_instruction=1
#task_file=tasks/peract_18_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val

#main_dir=hiveformer
#use_instruction=0
#task_file=tasks/hiveformer_74_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val

# Single-task PerAct
#train_iters=400_000
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb_fair.sh \
#   --tasks task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --train_iters $train_iters \
#   --variations {0..199} \
#   --run_log_dir $task-PERACT
#done

# Single-task HiveFormer
#main_dir=hiveformer_10_episodes
#train_iters=400_000
#max_episodes_per_task=10
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb_fair.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --train_iters $train_iters \
#   --max_episodes_per_task $max_episodes_per_task \
#   --run_log_dir $task-HIVEFORMER-10-episodes
#done

# Multi-task PerAct
#train_iters=4_000_000
#sbatch train_1gpu_32gb_fair.sh \
#   --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#   --tasks $(cat $task_file | tr '\n' ' ') \
#   --batch_size 32 \
#   --batch_size_val 8 \
#   --num_workers 16 \
#   --cache_size 0 \
#   --cache_size_val 0 \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --train_iters $train_iters \
#   --variations {0..199} \
#   --run_log_dir MULTI-TASK-PERACT

# Single-task HiveFormer 10 demos
main_dir=hiveformer_10_episodes
use_instruction=0
train_iters=400_000
task_file=tasks/hiveformer_74_tasks.csv
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val
batch_size=16
batch_size_val=4
num_workers=1
max_episodes_per_task=10
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb_fair.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --batch_size $batch_size \
   --batch_size_val $batch_size_val \
   --num_workers $num_workers \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --train_iters $train_iters \
   --max_episodes_per_task $max_episodes_per_task \
   --run_log_dir $task-HIVEFORMER-10-episodes
done
