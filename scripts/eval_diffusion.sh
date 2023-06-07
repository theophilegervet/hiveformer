exp=diffusion
ckpts=(
  unplug_charger
)
tasks=(
  unplug_charger
)

data_dir=/home/sirdome/katefgroup/datasets/raw/diffusion_trajectories_val
#data_dir=/home/sirdome/katefgroup/datasets/raw/diffusion_trajectories_train
num_episodes=5
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
use_instruction=1
num_ghost_points=10000
headless=0
offline=1
record_videos=1
max_tries=10
verbose=1

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth --model diffusion --verbose $verbose \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries $max_tries \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points
done
