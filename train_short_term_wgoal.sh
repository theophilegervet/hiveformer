dataset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_train/
valset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_val/
instructions=/private/home/theop123/hiveformer/instructions.pkl
main_dir=diffusion_short_term_wgoal
lr=2e-4
dense_interpolation=1
interpolation_length=50
B=6
ngpus=8
checkpoint=/private/home/theop123/hiveformer/train_logs/diffusion_short_term_wgoal/short_term_wgoal-B6-lr2e-4-short_term/last.pth

#CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node $ngpus --master_port $RANDOM main_trajectory.py \
sbatch train_8gpu_32gb_fair_devlab.sh \
    --tasks unplug_charger close_door open_box open_fridge put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups wipe_desk \
    --dataset  $dataset\
    --valset $valset \
    --instructions $instructions \
    --gripper_loc_bounds 10_tough_diffusion_location_bounds.json \
    --num_workers 4\
    --train_iters 1000000 \
    --model diffusion \
    --num_query_cross_attn_layers 2 \
    --feat_scales_to_use 3 \
    --embedding_dim 120 \
    --weight_tying 1 \
    --use_instruction 1 \
    --use_goal 1 \
    --use_goal_at_test 0 \
    --action_dim 8 \
    --val_freq 1000 \
    --predict_short 16 \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 12 \
    --cache_size 0 \
    --cache_size_val 0 \
    --lr $lr\
    --checkpoint $checkpoint \
    --run_log_dir short_term_wgoal-B$B-lr$lr-short_term
