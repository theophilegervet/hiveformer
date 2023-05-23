#!/bin/sh

main_dir=peract_new_data
use_instruction=1
train_iters=4_000_000
num_workers=5
cameras=left_shoulder,right_shoulder,wrist,front
task_file=tasks/peract_18_tasks.csv
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train_new
valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val_new

#main_dir=hiveformer
#use_instruction=0
#task_file=tasks/hiveformer_74_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val

# Single-task PerAct
#train_iters=400_000
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_2gpu_32gb_fair.sh \
#   --devices cuda:0 cuda:1 \
#   --tasks $task \
#   --cameras $cameras \
#   --dataset $dataset \
#   --valset $valset \
#   --num_workers 16 \
#   --cache_size 0 \
#   --cache_size_val 0 \
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

# Medium 1
batch_size_val=1
layers=2
embedding_dim=120
num_workers=1
batch_size=3
#checkpoint=/private/home/theop123/hiveformer2/train_logs/peract_new_data/PERACT-DDP-MULTI-TASK-120-2-1-6_version8373980/model.step=110000-value=0.00000.pth
sbatch train_8gpu_32gb_fair_devlab.sh \
   --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
   --tasks $(cat $task_file | tr '\n' ' ') \
   --cameras $cameras \
   --embedding_dim $embedding_dim \
   --batch_size $batch_size \
   --batch_size_val $batch_size_val \
   --num_workers $num_workers \
   --cache_size 0 \
   --cache_size_val 0 \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --train_iters $train_iters \
   --variations {0..199} \
   --num_ghost_point_cross_attn_layers $layers \
   --num_query_cross_attn_layers $layers \
   --num_vis_ins_attn_layers $layers \
   --run_log_dir PERACT-DDP-MULTI-TASK-$embedding_dim-$layers-$num_workers-$batch_size

# Medium 2
#batch_size=3
#batch_size_val=1
#layers=4
#embedding_dim=120
#sbatch train_8gpu_32gb_fair.sh \
#   --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
#   --tasks $(cat $task_file | tr '\n' ' ') \
#   --cameras $cameras \
#   --embedding_dim $embedding_dim \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size_val \
#   --num_workers $num_workers \
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
#   --num_ghost_point_cross_attn_layers $layers \
#   --num_query_cross_attn_layers $layers \
#   --num_vis_ins_attn_layers $layers \
#   --run_log_dir PERACT-MULTI-TASK-DDP-MEDIUM2-DISTRUBUTED-SAMPLER

# Big
#batch_size=2
#batch_size_val=1
#layers=4
#embedding_dim=240
#sbatch train_8gpu_32gb_fair.sh \
#   --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
#   --tasks $(cat $task_file | tr '\n' ' ') \
#   --cameras $cameras \
#   --embedding_dim $embedding_dim \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size_val \
#   --num_workers $num_workers \
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
#   --num_ghost_point_cross_attn_layers $layers \
#   --num_query_cross_attn_layers $layers \
#   --num_vis_ins_attn_layers $layers \
#   --run_log_dir PERACT-MULTI-TASK-DDP-BIG-DISTRUBUTED-SAMPLER
