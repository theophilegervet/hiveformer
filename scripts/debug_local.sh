exp=peract_new_data
ckpts=(
  # turn_tap-PERACT_version8003386
  # open_drawer-PERACT_version8003435
  # push_buttons-PERACT_version8003438
  sweep_to_dustpan_of_size-PERACT_version8003437
  # slide_block_to_color_target-PERACT_version8003438
  # insert_onto_square_peg-PERACT_version8003439
  # meat_off_grill-PERACT_version8003440
  # place_shape_in_shape_sorter-PERACT_version8003441
  # place_wine_at_rack_location-PERACT_version8003443
  # put_groceries_in_cupboard-PERACT_version8003443
  # put_money_in_safe-PERACT_version8003444
  # close_jar-PERACT_version8003445
  # reach_and_drag-PERACT_version8003446
  # light_bulb_in-PERACT_version8003447
  # stack_cups-PERACT_version8003448
  # place_cups-PERACT_version8003449
  # put_item_in_drawer-PERACT_version8003450
  # stack_blocks-PERACT_version8003451
)
tasks=(
  # turn_tap
  # open_drawer
  # push_buttons
  sweep_to_dustpan_of_size
  # slide_block_to_color_target
  # insert_onto_square_peg
  # meat_off_grill
  # place_shape_in_shape_sorter
  # place_wine_at_rack_location
  # put_groceries_in_cupboard
  # put_money_in_safe
  # close_jar
  # reach_and_drag
  # light_bulb_in
  # stack_cups
  # place_cups
  # put_item_in_drawer
  # stack_blocks
)

# data_dir=/home/sirdome/katefgroup/datasets/raw/18_peract_tasks_val_new
# # data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
# num_episodes=10
# # gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
# gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
# use_instruction=0
# # use_instruction=1
# num_ghost_points=10000
# headless=0
# offline=1
# record_videos=0
# max_tries=10
# max_steps=10

# num_ckpts=${#ckpts[@]}
# for ((i=0; i<$num_ckpts; i++)); do
#   python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#     --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
#     --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries $max_tries \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file --max_steps $max_steps --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
#     --variations {0..60}
# done

data_dir=/home/sirdome/katefgroup/datasets/raw/18_peract_tasks_val_new
num_episodes=10
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#use_instruction=0
use_instruction=1
num_ghost_points=10000
headless=0
offline=0
cameras="left_shoulder,right_shoulder,wrist,front"

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --variations {0..60} --cameras $cameras
done