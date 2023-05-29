dataset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_train/
valset=/private/home/theop123/datasets/rlbench/packaged/diffusion_trajectories_val/

main_dir=multitask_nolang


# task=unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up
lr=2e-4
dense_interpolation=1
interpolation_length=100
batch_size_val=3
train_iters=4_000_000

for batch_size in 12 24; do
  for num_workers in 1 4; do
    sbatch train_8gpu_32gb_fair.sh \
        --tasks unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
        --dataset  $dataset\
        --valset $valset \
        --gripper_loc_bounds_file diffusion_location_bounds.json \
        --num_workers $num_workers \
        --train_iters $train_iters \
        --use_instruction 1 \
        --use_rgb 1 \
        --use_goal 1 \
        --val_freq 1000 \
        --checkpoint_freq 5 \
        --dense_interpolation $dense_interpolation \
        --interpolation_length $interpolation_length \
        --gripper_bounds_buffer 0.02\
        --exp_log_dir $main_dir \
        --batch_size $batch_size \
        --batch_size_val $batch_size_val \
        --lr $lr\
        --logger wandb \
        --run_log_dir DIFFUSION-multitask-B$B-lr$lr-DI$dense_interpolation-$interpolation_length
  done
done
