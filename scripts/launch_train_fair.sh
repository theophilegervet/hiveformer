#!/bin/sh

# -----------------------------------------------------------------------------------------
# Main experiments 100 demos
# -----------------------------------------------------------------------------------------

# Single-task HiveFormer 100 demos
#main_dir=hiveformer
#use_instruction=0
#train_iters=400_000
#cameras=left_shoulder,right_shoulder,wrist
#task_file=tasks/hiveformer_74_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val
#batch_size=16
#batch_size_val=4
#num_workers=1
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb_fair.sh \
#   --tasks $task \
#   --cameras $cameras \
#   --dataset $dataset \
#   --valset $valset \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size_val \
#   --num_workers $num_workers \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --train_iters $train_iters \
#   --run_log_dir $task-HIVEFORMER
#done

# Multi-task PerAct 100 demos
main_dir=peract_new_data
use_instruction=1
train_iters=4_000_000
cameras=left_shoulder,right_shoulder,wrist,front
task_file=tasks/peract_18_tasks.csv
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train_new
valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val_new
batch_size=6
batch_size_val=1
layers=2
embedding_dim=120
num_workers=1
# 120K + 120K + 150K + 235K + 230K + ??
checkpoint=/private/home/theop123/hiveformer2/train_logs/peract_new_data/PERACT-MULTI-TASK_version8846643/model.step=230000-value=0.00000.pth
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
   --checkpoint $checkpoint \
   --run_log_dir PERACT-MULTI-TASK

main_dir=peract_debug
use_instruction=1
train_iters=4_000_000
cameras=left_shoulder,right_shoulder,wrist,front
task_file=tasks/peract_18_tasks.csv
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train_new
valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val_new
batch_size=1
batch_size_val=1
layers=2
embedding_dim=60
num_workers=1
python train.py \
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
   --run_log_dir PERACT-DEBUG

# -----------------------------------------------------------------------------------------
# Main experiments 10 demos
# -----------------------------------------------------------------------------------------

# Single-task HiveFormer 10 demos
#main_dir=hiveformer_10_episodes
#use_instruction=0
#train_iters=400_000
#cameras=left_shoulder,right_shoulder,wrist
#task_file=tasks/hiveformer_74_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_train
#valset=/private/home/theop123/datasets/rlbench/packaged/74_hiveformer_tasks_val
#batch_size=16
#batch_size_val=4
#num_workers=1
#max_episodes_per_task=10
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb_fair.sh \
#   --tasks $task \
#   --cameras $cameras \
#   --dataset $dataset \
#   --valset $valset \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size_val \
#   --num_workers $num_workers \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --train_iters $train_iters \
#   --max_episodes_per_task $max_episodes_per_task \
#   --run_log_dir $task-HIVEFORMER-10-episodes
#done

# Multi-task PerAct 10 demos
#main_dir=peract_10_episodes
#use_instruction=1
#train_iters=4_000_000
#cameras=left_shoulder,right_shoulder,wrist,front
#task_file=tasks/peract_18_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_train_new
#valset=/private/home/theop123/datasets/rlbench/packaged/18_peract_tasks_val_new
#batch_size=6
#batch_size_val=1
#layers=2
#embedding_dim=120
#num_workers=1
#max_episodes_per_task=10
#checkpoint=/private/home/theop123/hiveformer2/train_logs/peract_10_episodes/PERACT-MULTI-TASK-10-episodes_version8806896/model.step=240000-value=0.00000.pth
#for point_cloud_rotate_yaw_range in 0.0; do
#  sbatch train_8gpu_32gb_fair_devlab.sh \
#     --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
#     --tasks $(cat $task_file | tr '\n' ' ') \
#     --cameras $cameras \
#     --embedding_dim $embedding_dim \
#     --batch_size $batch_size \
#     --batch_size_val $batch_size_val \
#     --num_workers $num_workers \
#     --cache_size 0 \
#     --cache_size_val 0 \
#     --dataset $dataset \
#     --valset $valset \
#     --exp_log_dir $main_dir \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file \
#     --use_instruction $use_instruction \
#     --max_episodes_per_task $max_episodes_per_task \
#     --logger wandb \
#     --train_iters $train_iters \
#     --variations {0..199} \
#     --num_ghost_point_cross_attn_layers $layers \
#     --num_query_cross_attn_layers $layers \
#     --num_vis_ins_attn_layers $layers \
#     --checkpoint $checkpoint \
#     --point_cloud_rotate_yaw_range $point_cloud_rotate_yaw_range \
#     --run_log_dir PERACT-MULTI-TASK-10-episodes-$point_cloud_rotate_yaw_range
#done
