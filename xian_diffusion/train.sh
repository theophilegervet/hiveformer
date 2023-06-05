
dataset=/home/zhouxian/git/datasets/packaged/diffusion_trajectories_train/
valset=/home/zhouxian/git/datasets/packaged/diffusion_trajectories_train/

dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

# dataset=/scratch/rlbench/diffusion_trajectories_train/
# valset=/scratch/rlbench/diffusion_trajectories_val/

main_dir=diffuse_05_24
main_dir=diffuse_05_25
main_dir=diffuse_05_28
main_dir=diffuse_05_29
main_dir=diffuse_05_30
main_dir=diffuse_05_31_multitask
main_dir=diffuse_06_02_multitask
main_dir=diffuse_06_05

# main_dir=debug
# task_file=tasks/diffusion_10_tough_tasks.csv

# task=close_door
# task=wipe_desk
# task=10_tough_tasks
task=hang_frame_on_hanger
bound_file=12_tough_diffusion_location_bounds.json

lr=1e-4
dense_interpolation=1
interpolation_length=50
B=24
n_gpus=4
use_instruction=1
num_query_cross_attn_layers=4

B_gpu=$((B/n_gpus))
python train_diffusion.py \
    --master_port 29500\
    --tasks $task\
    --n_gpus $n_gpus\
    --dataset $dataset\
    --valset $valset \
    --instructions instructions_old/instructions_local.pkl \
    --gripper_loc_bounds_file $bound_file\
    --use_instruction $use_instruction \
    --num_workers 4\
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
    --run_log_dir $task-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-L$num_query_cross_attn_layers\
    # --checkpoint train_logs/diffuse_06_02_multitask/10_tough_tasks-B24-lr1e-4-DI1-50-L4/last.pth\
    
