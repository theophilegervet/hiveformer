
dataset=/home/zhouxian/git/datasets/packaged/diffusion_trajectories_train/
valset=/home/zhouxian/git/datasets/packaged/diffusion_trajectories_val/

dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

main_dir=diffuse_05_24
main_dir=diffuse_05_25
main_dir=diffuse_05_28
main_dir=diffuse_05_29
main_dir=diffuse_05_30

# main_dir=debug

task=close_door
task=wipe_desk

lr=1e-4
dense_interpolation=1
interpolation_length=50
predict_length=0
denoise_steps=50
B=24

num_query_cross_attn_layers=2

python train_diffusion.py \
    --tasks $task \
    --dataset  $dataset\
    --valset $valset \
    --instructions instructions_old/instructions_local.pkl \
Expand All
    @@ -29,9 +39,12 @@ python train_diffusion.py --tasks $task \
    --checkpoint_freq 5 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --predict_length $predict_length \
    --denoise_steps $denoise_steps\
    --num_query_cross_attn_layers $num_query_cross_attn_layers \
    --gripper_bounds_buffer 0.02\
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 12 \
    --lr $lr\
    --run_log_dir $task-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-PL$predict_length-L$num_query_cross_attn_layers-DN$denoise_steps\