valset=~/katefgroup/datasets/raw/diffusion_trajectories_val/
task=close_door
use_goal=1
use_instruction=0
gripper_bounds_buffer=0.02

dense_interpolation=1
interpolation_length=100
ckpt=/home/zhouxian/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-100_version174698/model.step=380000-value=0.00000.pth

dense_interpolation=1
interpolation_length=100
ckpt=regression_model.pth

python eval.py\
     --tasks $task\
     --checkpoint $ckpt\
     --data_dir $valset\
     --instructions instructions.pkl \
     --gripper_loc_bounds_file diffusion_location_bounds.json\
     --use_goal $use_goal \
     --model regression \
     --num_query_cross_attn_layers 2 \
     --dense_interpolation $dense_interpolation \
     --interpolation_length $interpolation_length \
     --offline 0\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --run_log_dir $task-ONLINE\
     --max_steps -1 \
     --max_tries 10
