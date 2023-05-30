exp=hiveformer_10_episodes
ckpts=(
  close_fridge-HIVEFORMER-10-episodes_version8815938
  unplug_charger-HIVEFORMER-10-episodes_version8815946
  play_jenga-HIVEFORMER-10-episodes_version8815958
  slide_cabinet_open_and_place_cups-HIVEFORMER-10-episodes_version8816006
  insert_onto_square_peg-HIVEFORMER-10-episodes_version8815982
  stack_wine-HIVEFORMER-10-episodes_version8816001
  pick_and_lift_small-HIVEFORMER-10-episodes_version8815971
  phone_on_base-HIVEFORMER-10-episodes_version8815987
  put_money_in_safe-HIVEFORMER-10-episodes_version8815992
  open_microwave-HIVEFORMER-10-episodes_version8815955
  place_shape_in_shape_sorter-HIVEFORMER-10-episodes_version8815989
  wipe_desk-HIVEFORMER-10-episodes_version8816003
  setup_checkers-HIVEFORMER-10-episodes_version8816000
  meat_off_grill-HIVEFORMER-10-episodes_version8815984
  open_box-HIVEFORMER-10-episodes_version8815950
  slide_block_to_target-HIVEFORMER-10-episodes_version8815944
  move_hanger-HIVEFORMER-10-episodes_version8815985
  open_drawer-HIVEFORMER-10-episodes_version8815952
  straighten_rope-HIVEFORMER-10-episodes_version8816004
  take_usb_out_of_computer-HIVEFORMER-10-episodes_version8815945
  sweep_to_dustpan-HIVEFORMER-10-episodes_version8815993
  insert_usb_in_computer-HIVEFORMER-10-episodes_version8815984
  lamp_on-HIVEFORMER-10-episodes_version8815948
  stack_blocks-HIVEFORMER-10-episodes_version8816009
  close_grill-HIVEFORMER-10-episodes_version8815966
  take_frame_off_hanger-HIVEFORMER-10-episodes_version8815976
  put_umbrella_in_umbrella_stand-HIVEFORMER-10-episodes_version8815974
  put_knife_on_chopping_board-HIVEFORMER-10-episodes_version8815972
  open_window-HIVEFORMER-10-episodes_version8815969","running
  change_clock-HIVEFORMER-10-episodes_version8815964
  pick_up_cup-HIVEFORMER-10-episodes_version8815957
  place_hanger_on_rack-HIVEFORMER-10-episodes_version8815988
  hang_frame_on_hanger-HIVEFORMER-10-episodes_version8815968
  pick_and_lift-HIVEFORMER-10-episodes_version8815970
  lift_numbered_block-HIVEFORMER-10-episodes_version8815949
  open_wine_bottle-HIVEFORMER-10-episodes_version8815956
  put_books_on_bookshelf-HIVEFORMER-10-episodes_version8815991
  close_door-HIVEFORMER-10-episodes_version8815947
  stack_cups-HIVEFORMER-10-episodes_version8816007
  reach_and_drag-HIVEFORMER-10-episodes_version8815997
  tv_on-HIVEFORMER-10-episodes_version8816005
  plug_charger_in_power_supply-HIVEFORMER-10-episodes_version8815990
  change_channel-HIVEFORMER-10-episodes_version8816004
  open_fridge-HIVEFORMER-10-episodes_version8815953
  take_toilet_roll_off_stand-HIVEFORMER-10-episodes_version8815979
  open_grill-HIVEFORMER-10-episodes_version8815954
  open_door-HIVEFORMER-10-episodes_version8815968
  meat_on_grill-HIVEFORMER-10-episodes_version8815987
  open_oven-HIVEFORMER-10-episodes_version8815986
  screw_nail-HIVEFORMER-10-episodes_version8815998
  take_plate_off_colored_dish_rack-HIVEFORMER-10-episodes_version8815994
  tower3-HIVEFORMER-10-episodes_version8816002
  water_plants-HIVEFORMER-10-episodes_version8815995
)

tasks=(
  close_fridge
  unplug_charger
  play_jenga
  slide_cabinet_open_and_place_cups
  insert_onto_square_peg
  stack_wine
  pick_and_lift_small
  phone_on_base
  put_money_in_safe
  open_microwave
  place_shape_in_shape_sorter
  wipe_desk
  setup_checkers
  meat_off_grill
  open_box
  slide_block_to_target
  move_hanger
  open_drawer
  straighten_rope
  take_usb_out_of_computer
  sweep_to_dustpan
  insert_usb_in_computer
  lamp_on
  stack_blocks
  close_grill
  take_frame_off_hanger
  put_umbrella_in_umbrella_stand
  put_knife_on_chopping_board
  open_window
  change_clock
  pick_up_cup
  place_hanger_on_rack
  hang_frame_on_hanger
  pick_and_lift
  lift_numbered_block
  open_wine_bottle
  put_books_on_bookshelf
  close_door
  stack_cups
  reach_and_drag
  tv_on
  plug_charger_in_power_supply
  change_channel
  open_fridge
  take_toilet_roll_off_stand
  open_grill
  open_door
  meat_on_grill
  open_oven
  screw_nail
  take_plate_off_colored_dish_rack
  tower3
  water_plants
)

data_dir=/home/sirdome/katefgroup/datasets/raw/74_hiveformer_tasks_val
#data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=50
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=0
#use_instruction=1
num_ghost_points=10000
headless=1
offline=0

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points
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
