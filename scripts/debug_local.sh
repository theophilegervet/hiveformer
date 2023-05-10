# HIVEFORMER
exp=03_24_hiveformer_setting
ckpts=(
#  insert_onto_square_peg-HIVEFORMER_version161659
  # move_hanger-HIVEFORMER_version161663
)
tasks=(
#  insert_onto_square_peg
  # move_hanger
)

exp=05_04_eval_on_peract_18_tasks
ckpts=(
  # insert_onto_square_peg-PERACT_version7598186
  close_jar-PERACT_version7453211
  # light_bulb_in-PERACT_version7598189
  # place_shape_in_shape_sorter-PERACT_version7598184
  # put_money_in_safe-PERACT_version7598187
  # put_groceries_in_cupboard-PERACT_version7453209
)
tasks=(
  # insert_onto_square_peg
  close_jar
  # light_bulb_in
  # place_shape_in_shape_sorter
  # put_money_in_safe
  # put_groceries_in_cupboard
)

# data_dir=/home/sirdome/katefgroup/raw/74_hiveformer_tasks_val
data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=100
# gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
# use_instruction=0
use_instruction=1
num_ghost_points=10000
headless=0
offline=1
record_videos=1
max_tries=10
max_steps=10

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction --max_tries $max_tries \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --max_steps $max_steps --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --variations {0..60}
done
