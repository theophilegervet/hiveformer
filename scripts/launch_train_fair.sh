#!/bin/sh

main_dir=05_04_eval_on_peract_18_tasks
use_instruction=1
task_file=tasks/peract_18_tasks.csv
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train
valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val
train_iters=200_000
#for task in $(cat $task_file | tr '\n' ' '); do
for task in place_wine_at_rack_location; do
  sbatch train_1gpu_32gb_fair.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --variations {0..199} \
   --train_iters $train_iters \
   --run_log_dir $task-PERACT
done
