# main_dir=02_27_rotation
# main_dir=02_28_rotation_0.1
# main_dir=02_28_rotation_0.1_quatloss
# main_dir=02_28_rotation_0.1_att
# main_dir=02_28_rotation_0.1_att_debug
# main_dir=02_28_rotation_0.1_att_nonepe
# main_dir=02_28_rotation_0.1_att_noatt
# main_dir=03_03_multi_level_sampling
# main_dir=03_04_multi_level_sampling
# main_dir=03_09_ablations
# main_dir=03_10_dense_val_sampling
# main_dir=03_13
# main_dir=03_14_debug_offset0
# main_dir=03_19_seed
# main_dir=03_19_compare
# main_dir=03_19_more_tasks
# main_dir=03_21_10demo
# main_dir=03_22_10demo_small
# main_dir=03_22_knife
# main_dir=03_22_wine
# main_dir=03_23
# main_dir=03_24
main_dir=04_05_multitask
main_dir=04_06_multitask_noinstr
main_dir=04_08_multitask_fixbug
main_dir=04_10_multitask_revert
main_dir=04_12_multitask
main_dir=04_13_multitask
main_dir=04_16_multitask_cont
main_dir=04_17_multitask_vis_ins_att_complex
main_dir=04_17_multitask_res422
main_dir=04_20_multitask
main_dir=04_22_multitask
# main_dir=debug

# dataset=/home/tgervet/datasets/hiveformer/packaged/2
# valset=/home/tgervet/datasets/hiveformer/packaged/3
# dataset=/home/zhouxian/git/datasets/hiveformer/packaged/2
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
# dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
# valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
dataset=/scratch/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
valset=/scratch/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
# dataset=/home/zhouxian/git/datasets/packaged/74_hiveformer_tasks_train
# valset=/home/zhouxian/git/datasets/packaged/74_hiveformer_tasks_val
# task=reach_target
# task=push_button
# task=slide_block_to_target
# task=pick_up_cup
# task=take_umbrella_out_of_umbrella_stand
# task=pick_and_lift
# task=put_knife_on_chopping_board
# task=take_money_out_safe
# task=put_money_in_safe
# task=stack_wine

# task_file=tasks/pick_and_lift.csv

task_file=tasks/10_autolambda_tasks.csv
task=10_tasks

num_workers=10
train_cache_size=0
val_cache_size=0
train_iters=1000000

batch_size_val=4
lr=1e-4

gripper_bounds_buffer=0.04
use_instruction=1
weight_tying=1
max_episodes_per_taskvar=100
num_ghost_points=1000
num_ghost_points_val=10000
simplify_ins=0
seed=0
embedding_dim=60
n_layer=2
num_sampling_level=3
gp_emb_tying=1
simplify=1

batch_size=16
symmetric_rotation_loss=0
ins_pos_emb=0
vis_ins_att=1
vis_ins_att_complex=1

regress_position_offset=0

     # --devices cuda:0 cuda:1\
     # --checkpoint /home/xianz1/git/hiveformer/train_logs/04_13_multitask/10_tasks-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo100-dim60-L2-lr1e-4-seed0-simpins0-ins_pos_emb1-vis_ins_att1_version164229/model.step=570000-value=0.00000.pth \

python train.py\
     --devices cuda:0 cuda:1\
     --tasks $(cat $task_file | tr '\n' ' ') \
     --dataset $dataset \
     --valset $valset \
     --instructions instructions_new.pkl \
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
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-T$num_ghost_points-V$num_ghost_points_val-symrot$symmetric_rotation_loss-gptie$gp_emb_tying-simp$simplify-B$batch_size-demo$max_episodes_per_taskvar-dim$embedding_dim-L$n_layer-lr$lr-seed$seed-simpins$simplify_ins-ins_pos_emb$ins_pos_emb-vis_ins_att$vis_ins_att-vis_ins_att_complex$vis_ins_att_complex
