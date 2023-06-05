
dataset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_train/
valset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_val/

main_dir=diffuse_06_03_multitask

task_file=tasks/diffusion_10_hiveformer_tasks.csv
task=10_hiveformer_tasks
bound_file=10_hiveformer_diffusion_location_bounds.json

lr=1e-4
dense_interpolation=1
interpolation_length=50
B=24
n_gpus=8
use_instruction=1
#num_query_cross_attn_layers=4
num_query_cross_attn_layers=8

B_gpu=$((B/n_gpus))
#python train_diffusion.py \
sbatch train_8gpu_32gb_fair_devlab.sh \
    --master_port 29500\
    --tasks $(cat $task_file | tr '\n' ' ')\
    --n_gpus $n_gpus\
    --dataset $dataset\
    --valset $valset \
    --instructions instructions.pkl \
    --gripper_loc_bounds_file $bound_file\
    --use_instruction $use_instruction \
    --num_workers 8\
    --train_iters 1000000\
    --use_rgb 1 \
    --use_goal 1 \
    --cache_size 0 \
    --cache_size_val 0 \
    --val_freq 2000 \
    --checkpoint_freq 1 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --num_query_cross_attn_layers $num_query_cross_attn_layers \
    --gripper_bounds_buffer 0.02\
    --exp_log_dir $main_dir \
    --batch_size $B_gpu \
    --batch_size_val 12 \
    --lr $lr\
    --logger wandb\
    --run_log_dir $task-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-L$num_query_cross_attn_layers
    
