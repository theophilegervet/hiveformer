# HIVEFORMER
#exp=03_24_hiveformer_setting
#ckpts=(
#  open_fridge-HIVEFORMER_version161072
#  hang_frame_on_hanger-HIVEFORMER_version161481
#  open_door-HIVEFORMER_version161482
#  take_frame_off_hanger-HIVEFORMER_version161654
#  open_oven-HIVEFORMER_version161968
#  put_books_on_bookshelf-HIVEFORMER_version161973
#  straighten_rope-HIVEFORMER_version163672
#  change_channel-HIVEFORMER_version165805
#  tv_on-HIVEFORMER_version164906
#  slide_cabinet_open_and_place_cups-HIVEFORMER_version165804
#  stack_cups-HIVEFORMER_version7658919
#  stack_blocks-HIVEFORMER_version7658922
#)
#tasks=(
#  open_fridge
#  hang_frame_on_hanger
#  open_door
#  take_frame_off_hanger
#  open_oven
#  put_books_on_bookshelf
#  straighten_rope
#  change_channel
#  tv_on
#  slide_cabinet_open_and_place_cups
#  stack_cups
#  stack_blocks
#)

# PERACT
exp=peract_new_data
ckpts=(
  multi-task2
  multi-task2
#  multi-task2
#  multi-task2

#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
#  multi-task2
)
tasks=(
  close_jar
  put_item_in_drawer
#  put_groceries_in_cupboard
#  put_money_in_safe

#  turn_tap
#  open_drawer
#  push_buttons
#  sweep_to_dustpan_of_size
#  slide_block_to_color_target
#  insert_onto_square_peg
#  meat_off_grill
#  place_shape_in_shape_sorter
#  place_wine_at_rack_location
#  put_groceries_in_cupboard
#  put_money_in_safe
#  close_jar
#  reach_and_drag
#  light_bulb_in
#  stack_cups
#  place_cups
#  put_item_in_drawer
#  stack_blocks
)

#data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val
#data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
#data_dir=/home/sirdome/katefgroup/datasets/raw/18_peract_tasks_val_new
data_dir=/home/katefgroup/Documents/datasets/rlbench/raw/18_peract_tasks_val_new
num_episodes=3
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#use_instruction=0
use_instruction=1
num_ghost_points=10000
headless=0
offline=0
record_videos=1
cameras="left_shoulder,right_shoulder,wrist,front"
embedding_dim=120

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos $record_videos --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --variations {0..60} --cameras $cameras --embedding_dim $embedding_dim
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
