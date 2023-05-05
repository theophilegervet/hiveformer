#!/bin/sh

#main_dir=05_04_eval_on_peract_18_tasks
#use_instruction=1
#task_file=tasks/peract_18_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val
#train_iters=400_000
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb_fair.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --variations {0..199} \
#   --train_iters $train_iters \
#   --run_log_dir $task-PERACT
#done

main_dir=05_05_eval_6d_rotation
use_instruction=0
task_file=tasks/hiveformer_high_precision_tasks.csv
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val
train_iters=400_000
for task in $(cat $task_file | tr '\n' ' '); do
  for rotation_parametrization in 6D_from_query; do
    for rotation_loss_coeff in 1.0; do
      sbatch train_1gpu_32gb_fair.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --exp_log_dir $main_dir \
       --gripper_loc_bounds_file $gripper_loc_bounds_file \
       --use_instruction $use_instruction \
       --logger wandb \
       --train_iters $train_iters \
       --run_log_dir $task-HIVEFORMER-$rotation_parametrization
    done
  done
done
