#!/bin/sh

#main_dir=peract
#use_instruction=1
#task_file=tasks/peract_18_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val

main_dir=hiveformer
use_instruction=0
task_file=tasks/hiveformer_74_tasks.csv
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val

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
main_dir=hiveformer_512x512
task_file=tasks/hiveformer_high_precision_tasks.csv
dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train_512x512
valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val_512x512
train_iters=800_000
image_size="512,512"
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb_128gb_fair.sh \
   --tasks $task \
   --image_size $image_size \
   --batch_size 4 \
   --batch_size_val 1 \
   --cache_size_val 0 \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --train_iters $train_iters \
   --run_log_dir $task-HIVEFORMER-512x512
done

# Multi-task
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
