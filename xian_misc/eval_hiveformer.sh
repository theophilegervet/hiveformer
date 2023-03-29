valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val/
# valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val_fail/
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
task=reach_target
task=push_button
task=slide_block_to_target
task=pick_up_cup
task=take_umbrella_out_of_umbrella_stand
task=pick_and_lift
# task=put_knife_on_chopping_board
# task=take_money_out_safe
# task=put_money_in_safe
# task=stack_wine


python eval.py\
     --tasks $task\
     --checkpoint /home/zhouxian/git/hiveformer/train_logs/03_25_hiveformer/pick_and_lift-demo100_version161031/model.step=100000-value=0.00000.pth \
     --data_dir $valset\
     --image_size 128,128\
     --offline 0\
     --model original\
     --num_episodes 100\
     --run_log_dir $task-ONLINE\
     --randomize_vp 1 \
     