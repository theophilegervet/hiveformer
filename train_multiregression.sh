dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

main_dir=multiregression_hard12


# task=unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up
lr=1e-4
dense_interpolation=1
interpolation_length=50
B=12

python train_diffusion.py --tasks unplug_charger close_door open_box open_fridge hang_frame_on_hanger put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf wipe_desk slide_cabinet_open_and_place_cups take_shoes_out_of_box \
    --master_port $RANDOM \
    --n_gpus 2\
    --dataset $dataset\
    --valset $valset \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --gripper_loc_bounds_file 12_tough_diffusion_location_bounds.json \
    --num_workers 4\
    --train_iters 500000\
    --use_instruction 1 \
    --model regression \
    --use_rgb 1 \
    --use_goal 1 \
    --cache_size 0 \
    --cache_size_val 0 \
    --num_query_cross_attn_layers 2 \
    --val_freq 1000 \
    --checkpoint_freq 5 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --gripper_bounds_buffer 0.02\
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 12 \
    --lr $lr\
    --run_log_dir multiregression_12hard-B$B-lr$lr-DI$dense_interpolation-$interpolation_length
