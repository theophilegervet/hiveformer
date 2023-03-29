
main_dir=03_25_hiveformer

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
# dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train_newkeyframe
# valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val_newkeyframe
task=reach_target
task=push_button
task=slide_block_to_target
task=pick_up_cup
task=take_umbrella_out_of_umbrella_stand
task=pick_and_lift
task=put_knife_on_chopping_board
task=take_money_out_safe
task=put_money_in_safe
# task=stack_wine

max_episodes_per_taskvar=100

python train.py \
     --tasks $task \
     --image_size 128,128\
     --dataset $dataset \
     --valset $valset \
     --model original\
     --exp_log_dir $main_dir \
     --max_episodes_per_taskvar $max_episodes_per_taskvar\
     --run_log_dir $task-demo$max_episodes_per_taskvar