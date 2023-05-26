python train_diffusion.py --tasks close_door \
    --dataset /home/zhouxian/git/datasets/packaged/diffusion_trajectories_train/ \
    --valset /home/zhouxian/git/datasets/packaged/diffusion_trajectories_val/ \
    --instructions instructions_new.pkl \
    --gripper_loc_bounds_file diffusion_location_bounds.json\
    --use_instruction 0 \
    --use_goal 1 \
    --val_freq 1000 \
    --checkpoint_freq 5 \
    --gripper_bounds_buffer 0.02\
    --exp_log_dir diffuse_05_24 \
    --run_log_dir close_door \
    --batch_size 24 --batch_size_val 12 --lr 1e-4

# python train_diffusion.py --tasks unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up \
#     --dataset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/ \
#     --valset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/ \
#     --instructions /home/tgervet/hiveformer/instructions.pkl \
#     --use_instruction 0 \
#     --use_goal 1 \
#     --exp_log_dir testm \
#     --batch_size 84 --batch_size_val 42 --lr 3e-4


# /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train