dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

main_dir=regress_06_30_multitask_2gpus


# task=unplug_charger close_door open_box open_fridge put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups wipe_desk
lr=1e-4
dense_interpolation=1
interpolation_length=50
B=12

CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 2 --master_port $RANDOM \
    main_trajectory.py --tasks unplug_charger close_door open_box open_fridge put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups wipe_desk \
    --dataset  $dataset\
    --valset $valset \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --gripper_loc_bounds 10_tough_diffusion_location_bounds.json \
    --num_workers 4\
    --train_iters 500000 \
    --model regression \
    --num_query_cross_attn_layers 2 \
    --feat_scales_to_use 3 \
    --embedding_dim 120 \
    --weight_tying 1 \
    --use_instruction 1 \
    --use_goal 1 \
    --action_dim 7 \
    --val_freq 1000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 12 \
    --cache_size 0 \
    --cache_size_val 0 \
    --lr $lr\
    --run_log_dir multi-B$B-lr$lr-DI$dense_interpolation-$interpolation_length \
    --checkpoint /home/ngkanats/analogical_manipulation/train_logs/regress_06_30_multitask_2gpus/multi-B12-lr1e-4-DI1-50/last.pth