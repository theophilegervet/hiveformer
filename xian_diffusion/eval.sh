valset=/home/zhouxian/git/datasets/raw/diffusion_trajectories_val/
valset=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val

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


task=close_door
task=wipe_desk
task=sweep_to_dustpan

offline=1


python eval.py\
     --seed 0\
     --tasks $task\
     --checkpoint $ckpt\
     --data_dir $valset\
     --instructions instructions.pkl \
     --gripper_loc_bounds_file $bound_file\
     --use_goal $use_goal \
     --model diffusion \
     --dense_interpolation $dense_interpolation \
     --num_query_cross_attn_layers $num_query_cross_attn_layers \
     --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
     --interpolation_length $interpolation_length \
     --offline $offline\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --run_log_dir $task-ONLINE\
     --max_steps -1 \
     --max_tries 10\
     --record_imgs 0