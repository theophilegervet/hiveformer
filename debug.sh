# python train_diffusion.py --tasks close_door \
#     --dataset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/ \
#     --valset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/ \
#     --instructions /home/tgervet/hiveformer/instructions.pkl \
#     --use_instruction 0 \
#     --use_goal 1 \
#     --exp_log_dir close_door_no_lang_no_goal \
#     --batch_size 48 --batch_size_val 24 --lr 3e-4 \
#     --val_freq 1 \
#     --checkpoint /home/ngkanats/hiveformer/train_logs/close_door_no_lang_no_goal/run_version173565/best.pth

python train_diffusion.py --tasks unplug_charger close_door open_box open_drawer open_fridge open_door put_umbrella_in_umbrella_stand take_frame_off_hanger open_oven put_books_on_bookshelf slide_cabinet_open_and_place_cups put_knife_on_chopping_board wipe_desk reach_target pick_up_cup stack_cups stack_blocks open_grill open_microwave toilet_seat_up \
    --dataset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/ \
    --valset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/ \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --use_instruction 0 \
    --use_goal 0 \
    --diffusion_head simple --use_rgb 0 \
    --exp_log_dir test \
    --trim_to_fixed_len 16 \
    --batch_size 32 --batch_size_val 42 --lr 1e-4 \
    --num_workers 4 \
    --checkpoint /home/ngkanats/hiveformer/train_logs/test/run_version173809/best.pth


# /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train