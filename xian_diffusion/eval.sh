valset=/home/zhouxian/git/datasets/raw/diffusion_trajectories_val/
task=close_door
use_goal=1
use_instruction=0
gripper_bounds_buffer=0.02

dense_interpolation=0
interpolation_length=100

python eval.py\
     --tasks $task\
     --checkpoint /home/zhouxian/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-50_version174697/model.step=100000-value=0.00000.pth \
     --data_dir $valset\
     --instructions instructions_new.pkl \
     --gripper_loc_bounds_file diffusion_location_bounds.json\
     --use_goal $use_goal \
     --model diffusion \
     --dense_interpolation $dense_interpolation \
     --interpolation_length $interpolation_length \
     --offline 0\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --run_log_dir $task-ONLINE\
     --max_steps -1 \
     --max_tries 10