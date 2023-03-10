
# main_dir=02_27_rotation
# main_dir=02_28_rotation_0.1
# main_dir=02_28_rotation_0.1_quatloss
# main_dir=02_28_rotation_0.1_att
# main_dir=02_28_rotation_0.1_att_debug
# main_dir=02_28_rotation_0.1_att_nonepe
# main_dir=02_28_rotation_0.1_att_noatt
# main_dir=03_03_multi_level_sampling
# main_dir=03_04_multi_level_sampling
main_dir=03_08_ablations

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
# dataset=/home/zhouxian/git/datasets/hiveformer/packaged/2
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
task=put_money_in_safe
batch_size=16
# batch_size=8
# batch_size=4
lr=1e-4

gripper_bounds_buffer=0.04
use_instruction=0
weight_tying=1
max_episodes_per_taskvar=100

num_sampling_level=4
regress_position_offset=0
num_ghost_points=1000
symmetric_rotation_loss=0

python train.py\
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --weight_tying $weight_tying\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --max_episodes_per_taskvar $max_episodes_per_taskvar\
     --symmetric_rotation_loss $symmetric_rotation_loss\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --lr $lr\
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-P$num_ghost_points-symrot$symmetric_rotation_loss-B$batch_size-lr$lr
