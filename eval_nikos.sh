exp=diffusion

tasks=(
#   close_door
#   hang_frame_on_hanger
#   open_box
#   open_fridge
#   open_oven
#   put_books_on_bookshelf
#   put_umbrella_in_umbrella_stand
#   stack_cups
#   take_frame_off_hanger
#   wipe_desk
#     take_shoes_out_of_box
    slide_cabinet_open_and_place_cups
#     unplug_charger

)
tasks=(
     place_wine_at_rack_location
)

data_dir=/home/sirdome/katefgroup/datasets/raw/74_hiveformer_tasks_val/
data_dir=/home/sirdome/katefgroup/datasets/raw/18_peract_tasks_val_new/
#data_dir=/home/sirdome/katefgroup/datasets/raw/diffusion_trajectories_train
num_episodes=32
gripper_loc_bounds_file=12_tough_diffusion_location_bounds.json
use_instruction=1
num_ghost_points=10000
headless=0
offline=1
record_videos=0
max_tries=10
verbose=1
interpolation_length=50

# num_ckpts=${#tasks[@]}
# for ((i=0; i<$num_ckpts; i++)); do
#   CUDA_LAUNCH_BLOCKING=1 python eval.py --tasks ${tasks[$i]} --act3d_checkpoint checkpoints/${tasks[$i]}.pth --model diffusion --traj_model diffusion --verbose $verbose \
#     --checkpoint checkpoints/multiregression_4layers.pth --num_query_cross_attn_layers 4 \
#     --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
#     --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries $max_tries \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
#     --interpolation_length $interpolation_length --dense_interpolation 1
# done

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
  CUDA_LAUNCH_BLOCKING=1 python eval.py --tasks ${tasks[$i]} --act3d_checkpoint checkpoints/slide_cabinet_open_and_place_cups.pth --model diffusion --traj_model diffusion --verbose $verbose \
    --checkpoint 10-hiveformer-last.pth --num_query_cross_attn_layers 4 \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries $max_tries \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --interpolation_length $interpolation_length --dense_interpolation 1
done
# num_ckpts=${#tasks[@]}
# for ((i=0; i<$num_ckpts; i++)); do
#   CUDA_LAUNCH_BLOCKING=1 python eval.py --tasks ${tasks[$i]} --act3d_checkpoint checkpoints/${tasks[$i]}.pth --model diffusion --traj_model diffusion --verbose $verbose \
#     --checkpoint checkpoints/multiwhole_2layers.pth --num_query_cross_attn_layers 2 \
#     --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
#     --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries 1 \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
#     --interpolation_length 300 --dense_interpolation 1 --use_goal_at_test 0 --action_dim 8
# done