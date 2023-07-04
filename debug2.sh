dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

main_dir=diffuse_06_21


# task=unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up
lr=1e-4
dense_interpolation=1
interpolation_length=50
B=24

CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
main_trajectory.py --tasks reach_target \
    --dataset  $dataset\
    --valset $valset \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --num_workers 4\
    --train_iters 500000 \
    --use_instruction 0 \
    --use_rgb 1 \
    --use_goal 0 --use_goal_at_test 0 --action_dim 7 \
    --val_freq 1000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 24 \
    --lr $lr\
    --run_log_dir reach_target-goalless-B$B-lr$lr-DI$dense_interpolation-$interpolation_length \
    --checkpoint /home/ngkanats/analogical_manipulation/train_logs/diffuse_06_21/reach_target-goalless-B24-lr1e-4-DI1-50/last.pth