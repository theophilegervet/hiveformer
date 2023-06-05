valset=/home/zhouxian/git/datasets/raw/diffusion_trajectories_val/
valset=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val/

model=diffusion
use_goal=1
use_instruction=0
gripper_bounds_buffer=0.02
predict_length=0
single_task_gripper_loc_bounds=1
offline=0
num_query_cross_attn_layers=2
bound_file=diffusion_location_bounds.json

task=close_door

dense_interpolation=1
interpolation_length=100
ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-100_version174698/model.step=180000-value=0.00000.pth

dense_interpolation=1
interpolation_length=50
ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-50_version174697/model.step=415000-value=0.00000.pth

# predict_length=1
# dense_interpolation=0
# ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_05_28/close_door-B24-lr1e-4-DI0-100-PL1/best.pth

single_task_gripper_loc_bounds=0

task=wipe_desk
dense_interpolation=1
interpolation_length=50
ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_05_28/wipe_desk-B24-lr1e-4-DI1-50-PL0/last.pth

###############################################################
task=put_books_on_bookshelf
task=put_umbrella_in_umbrella_stand
task=unplug_charger
task=close_door
task=open_box
task=open_fridge
task=hang_frame_on_hanger
task=take_frame_off_hanger
task=open_oven
task=wipe_desk
task=slide_cabinet_open_and_place_cups

offline=0
model=full

dense_interpolation=1
interpolation_length=50
use_instruction=1
num_query_cross_attn_layers=4
ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_06_02_multitask/10_tough_tasks-B24-lr1e-4-DI1-50-L4/last.pth
bound_file=10_tough_diffusion_location_bounds.json

num_ghost_points_val=20000
act3d_bound_file=tasks/74_hiveformer_tasks_location_bounds.json
act3d_gripper_bounds_buffer=0.04
act3d_num_query_cross_attn_layers=2
act3d_use_instruction=0
act3d_ckpt=/home/zhouxian/Downloads/theo_act3d/slide_cabinet_open_and_place_cups-HIVEFORMER_version165804/model.step=195000-value=0.00000.pth
###############################################################

###############################################################
task=reach_target
task=push_button
task=put_knife_on_chopping_board
task=take_money_out_safe
task=stack_wine


offline=0
model=full

dense_interpolation=1
interpolation_length=50
use_instruction=1
num_query_cross_attn_layers=4
ckpt=/home/zhouxian/Downloads/10-hiveformer-last.pth
bound_file=10_hiveformer_diffusion_location_bounds.json

num_ghost_points_val=20000
act3d_bound_file=tasks/74_hiveformer_tasks_location_bounds.json
act3d_gripper_bounds_buffer=0.04
act3d_num_query_cross_attn_layers=2
act3d_use_instruction=0
act3d_ckpt=/home/zhouxian/Downloads/theo_act3d/slide_cabinet_open_and_place_cups-HIVEFORMER_version165804/model.step=195000-value=0.00000.pth
###############################################################

python eval.py\
     --seed 0\
     --tasks $task\
     --checkpoint $ckpt\
     --act3d_checkpoint $act3d_ckpt\
     --data_dir $valset\
     --instructions instructions.pkl \
     --gripper_loc_bounds_file $bound_file\
     --act3d_gripper_loc_bounds_file $act3d_bound_file\
     --use_goal $use_goal \
     --model $model \
     --dense_interpolation $dense_interpolation \
     --num_query_cross_attn_layers $num_query_cross_attn_layers \
     --act3d_num_query_cross_attn_layers $act3d_num_query_cross_attn_layers \
     --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
     --interpolation_length $interpolation_length \
     --offline $offline\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --act3d_use_instruction $act3d_use_instruction\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --act3d_gripper_bounds_buffer $act3d_gripper_bounds_buffer\
     --run_log_dir $task-ONLINE\
     --max_steps -1 \
     --max_tries 1