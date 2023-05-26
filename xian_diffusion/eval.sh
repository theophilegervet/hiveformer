valset=/home/zhouxian/git/datasets/raw/diffusion_trajectories_val/
task=close_door
use_goal=1
use_instruction=0
gripper_bounds_buffer=0.02

python eval.py\
     --tasks $task\
     --checkpoint /home/zhouxian/git/hiveformer/train_logs/diffuse_05_24/close_door_version0/model.step=180000-value=0.00000.pth \
     --data_dir $valset\
     --instructions instructions_new.pkl \
     --gripper_loc_bounds_file diffusion_location_bounds.json\
     --use_goal $use_goal \
     --model diffusion \
     --offline 1\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --run_log_dir $task-ONLINE\
     --max_steps -1 \
     --max_tries 10