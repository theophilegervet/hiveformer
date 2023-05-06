# HIVEFORMER
#exp=03_24_hiveformer_setting
#ckpts=(
#  take_usb_out_of_computer-HIVEFORMER_version163526
#  place_shape_in_shape_sorter-HIVEFORMER_version163673
#  sweep_to_dustpan-HIVEFORMER_version163674
#  take_plate_off_colored_dish_rack-HIVEFORMER_version163675
#  place_hanger_on_rack-HIVEFORMER_version163676
#  plug_charger_in_power_supply-HIVEFORMER_version163667
#  reach_and_drag-HIVEFORMER_version163669
#  setup_checkers-HIVEFORMER_version163670
#  tower3-HIVEFORMER_version162809
#  straighten_rope-HIVEFORMER_version163672
#  wipe_desk-HIVEFORMER_version164479
#  change_channel-HIVEFORMER_version165805
#  tv_on-HIVEFORMER_version164906
#  slide_cabinet_open_and_place_cups-HIVEFORMER_version165804
#  stack_cups-HIVEFORMER_version164902
#  stack_blocks-HIVEFORMER_version164905
#)
#tasks=(
#  take_usb_out_of_computer
#  place_shape_in_shape_sorter
#  sweep_to_dustpan
#  take_plate_off_colored_dish_rack
#  place_hanger_on_rack
#  plug_charger_in_power_supply
#  reach_and_drag
#  setup_checkers
#  tower3
#  straighten_rope
#  wipe_desk
#  change_channel
#  tv_on
#  slide_cabinet_open_and_place_cups
#  stack_cups
#  stack_blocks
#)

# PERACT
exp=05_04_eval_on_peract_18_tasks
ckpts=(
  turn_tap-PERACT_version7453200
  open_drawer-PERACT_version7453201
  push_buttons-PERACT_version7453202
  sweep_to_dustpan_of_size-PERACT_version7453203
  slide_block_to_color_target-PERACT_version7453204  # TODO Not converged
  meat_off_grill-PERACT_version7453206  # TODO Not converged
  place_shape_in_shape_sorter-PERACT_version7453207  # TODO Not converged
  place_wine_at_rack_location-PERACT_version7453208  # TODO Not converged
  put_groceries_in_cupboard-PERACT_version7453209  # TODO Not converged
  close_jar-PERACT_version7453211  # TODO Not converged
)
tasks=(
  turn_tap
  open_drawer
  push_buttons
  sweep_to_dustpan_of_size
  slide_block_to_color_target
  meat_off_grill
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  put_groceries_in_cupboard
  close_jar
)

#data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val
data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=100
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#use_instruction=0
use_instruction=1
num_ghost_points=10000
headless=1
offline=0

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --variations {0..60}
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
