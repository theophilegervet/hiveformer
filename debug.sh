CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    main_trajectory.py --tasks close_door \
    --n_gpus 1 --num_workers 4 \
    --dataset /projects/katefgroup/datasets/rlbench/micro_trajectory/diffusion_trajectories_train/ \
    --valset /projects/katefgroup/datasets/rlbench/micro_trajectory/diffusion_trajectories_val/ \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --gripper_loc_bounds 10_tough_diffusion_location_bounds.json \
    --use_instruction 0 \
    --use_goal 0 --use_goal_at_test 0 --action_dim 8 \
    --exp_log_dir close_door_no_lang_short_term \
    --num_query_cross_attn_layers 2 \
    --val_freq 1000 \
    --batch_size 12 --batch_size_val 6 --lr 1e-4 \
    --cache_size 50 \
    --cache_size_val 0 \
    --predict_short 16 \
    --attn_rounds 2 --feat_scales_to_use 2 \
    --checkpoint /home/ngkanats/analogical_manipulation/train_logs/close_door_no_lang_short_term/run/last.pth

# python train_diffusion.py --tasks unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up \
#     --dataset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/ \
#     --valset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/ \
#     --instructions /home/tgervet/hiveformer/instructions.pkl \
#     --use_instruction 0 \
#     --use_goal 1 \
#     --exp_log_dir testm \
#     --batch_size 84 --batch_size_val 42 --lr 3e-4


# /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train