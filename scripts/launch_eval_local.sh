# HIVEFORMER
exp=hiveformer_512x512
ckpts=(
  insert_onto_square_peg-HIVEFORMER-512x512_version8030563
  insert_usb_in_computer-HIVEFORMER-512x512_version8030564
  screw_nail-HIVEFORMER-512x512_version8030569
  put_umbrella_in_umbrella_stand-HIVEFORMER-512x512_version8030562
  reach_and_drag-HIVEFORMER-512x512_version8030567
)
tasks=(
  insert_onto_square_peg
  insert_usb_in_computer
  screw_nail
  put_umbrella_in_umbrella_stand
  reach_and_drag
)

# PERACT
#exp=05_04_eval_on_peract_18_tasks
#ckpts=(
#  light_bulb_in-PERACT_version7598189
#  stack_cups-PERACT_version7642636
#  place_cups-PERACT_version7642637
#  put_item_in_drawer-PERACT_version7642638
#  stack_blocks-PERACT_version7642639
#  turn_tap-PERACT_version7453200
#  open_drawer-PERACT_version7453201
#  push_buttons-PERACT_version7453202
#  sweep_to_dustpan_of_size-PERACT_version7453203
#  slide_block_to_color_target-PERACT_version7453204
#  meat_off_grill-PERACT_version7453206
#  put_groceries_in_cupboard-PERACT_version7453209
#  close_jar-PERACT_version7453211
#  place_shape_in_shape_sorter-PERACT_version7598184
#  place_wine_at_rack_location-PERACT_version7598185
#  insert_onto_square_peg-PERACT_version7598186
#  put_money_in_safe-PERACT_version7598187
#  reach_and_drag-PERACT_version7598188
#)
#tasks=(
#  light_bulb_in
#  stack_cups
#  place_cups
#  put_item_in_drawer
#  stack_blocks
#  turn_tap
#  open_drawer
#  push_buttons
#  sweep_to_dustpan_of_size
#  slide_block_to_color_target
#  meat_off_grill
#  put_groceries_in_cupboard
#  close_jar
#  place_shape_in_shape_sorter
#  place_wine_at_rack_location
#  insert_onto_square_peg
#  put_money_in_safe
#  reach_and_drag
#)

data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val_512x512
#data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=100
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=0
#use_instruction=1
num_ghost_points=10000
headless=1
offline=0
image_size="512,512"

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction --image_size $image_size \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points # \
#    --variations {0..60}
done


#exp=03_13_hiveformer_setting
#ckpts=(
#  slide_block_to_target-HIVEFORMER-SETTING_version157318
#  put_money_in_safe-HIVEFORMER-SETTING_version157317
#  take_money_out_safe-HIVEFORMER-SETTING_version157319
#  take_umbrella_out_of_umbrella_stand-HIVEFORMER-SETTING_version157320
#  pick_and_lift-HIVEFORMER-SETTING_version157314
#  pick_up_cup-HIVEFORMER-SETTING_version157315
#  put_knife_on_chopping_board-HIVEFORMER-SETTING_version157316
#)
#tasks=(
#  slide_block_to_target
#  put_money_in_safe
#  take_money_out_safe
#  take_umbrella_out_of_umbrella_stand
#  pick_and_lift
#  pick_up_cup
#  put_knife_on_chopping_board
#)
#data_dir=/home/theophile_gervet_gmail_com/datasets/raw/10_hiveformer_tasks_val
##data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
#num_episodes=50
#
#num_ckpts=${#ckpts[@]}
#for ((i=0; i<$num_ckpts; i++)); do
#  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#    --data_dir $data_dir --offline 0 --num_episodes $num_episodes \
#    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction 0
#done


#exp=02_20_compare_hiveformer_and_baseline
#ckpts=(
#  HIVEFORMER-put_money_in_safe_version153380
#  HIVEFORMER-slide_block_to_target_version153382
#  HIVEFORMER-take_umbrella_out_of_umbrella_stand_version153385
#)
#tasks=(
#  put_money_in_safe
#  slide_block_to_target
#  take_umbrella_out_of_umbrella_stand
#)
#data_dir=/home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1
#image_size="128,128"
#num_episodes=10
#
#num_ckpts=${#ckpts[@]}
#for ((i=0; i<$num_ckpts; i++)); do
#  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#    --data_dir $data_dir --image_size $image_size --offline 0 --num_episodes $num_episodes \
#    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE-HIVEFORMER --model original
#done
