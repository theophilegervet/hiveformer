#!/bin/sh

main_dir=peract
use_instruction=1
task_file=tasks/peract_18_tasks.csv
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train
valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val

#main_dir=hiveformer
#use_instruction=0
#task_file=tasks/hiveformer_74_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val

# Single-task
train_iters=400_000
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb_fair.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --train_iters $train_iters \
   --run_log_dir $task-PERACT
done
