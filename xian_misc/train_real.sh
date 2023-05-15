main_dir=05_14_real
main_dir=05_15_real
# main_dir=debug

dataset=/home/zhouxian/git/datasets/packaged/real_tasks_train
valset=/home/zhouxian/git/datasets/packaged/real_tasks_val
num_workers=0
train_cache_size=0
val_cache_size=0

dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/real_tasks_train
valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/real_tasks_val
num_workers=2
train_cache_size=100
val_cache_size=100

train_iters=500000

task=real_reach_target
task=real_press_stapler
task=real_press_hand_san
task=real_put_fruits_in_bowl
task=real_stack_bowls
task=real_unscrew_bottle_cap
task=real_transfer_beans
task=real_put_duck_in_oven
task=real_spread_sand
task=real_wipe_coffee

batch_size_val=4
lr=1e-4

gripper_bounds_buffer=0.10
use_instruction=1
weight_tying=1
max_episodes_per_taskvar=100
num_ghost_points=1000
num_ghost_points_val=10000
simplify_ins=0
seed=0
embedding_dim=60
n_layer=2
gp_emb_tying=1
simplify=1

symmetric_rotation_loss=0
vis_ins_att_complex=0
vis_ins_att=1

regress_position_offset=0
ins_pos_emb=0
instruction_file=instructions_real.pkl

num_sampling_level=3
batch_size=16
batch_size_val=5


python train.py\
     --devices cuda:0\
     --image_size "240,360"\
     --instructions instructions_old/$instruction_file \
     --tasks $task \
     --dataset $dataset \
     --gripper_loc_bounds_file tasks/real_tasks_location_bounds.json\
     --val_freq 500 \
     --checkpoint_freq 20\
     --valset $valset \
     --train_cache_size $train_cache_size \
     --val_cache_size $val_cache_size \
     --train_iters $train_iters \
     --num_workers $num_workers \
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --simplify $simplify\
     --simplify_ins $simplify_ins\
     --ins_pos_emb $ins_pos_emb\
     --vis_ins_att $vis_ins_att\
     --vis_ins_att_complex $vis_ins_att_complex\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --batch_size_val $batch_size_val \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --num_ghost_points_val $num_ghost_points_val\
     --max_episodes_per_taskvar $max_episodes_per_taskvar\
     --symmetric_rotation_loss $symmetric_rotation_loss\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --embedding_dim $embedding_dim\
     --num_ghost_point_cross_attn_layers $n_layer\
     --num_query_cross_attn_layers $n_layer\
     --num_vis_ins_attn_layers $n_layer\
     --seed $seed\
     --lr $lr\
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-T$num_ghost_points-V$num_ghost_points_val-symrot$symmetric_rotation_loss-gptie$gp_emb_tying-simp$simplify-B$batch_size-demo$max_episodes_per_taskvar-dim$embedding_dim-L$n_layer-lr$lr-seed$seed-simpins$simplify_ins-ins_pos_emb$ins_pos_emb-vis_ins_att$vis_ins_att-vis_ins_att_complex$vis_ins_att_complex-ins$instruction_file

