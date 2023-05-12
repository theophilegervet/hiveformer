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
main_dir=04_28_singletask
main_dir=04_29_singletask
main_dir=04_30_singletask
main_dir=05_01_singletask
main_dir=05_02_singletask
main_dir=05_03_singletask
main_dir=debug

# dataset=/home/tgervet/datasets/hiveformer/packaged/2
# valset=/home/tgervet/datasets/hiveformer/packaged/3
# dataset=/home/zhouxian/git/datasets/hiveformer/packaged/2
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
num_workers=0
train_cache_size=100
val_cache_size=100

# dataset=/home/zhouxian/git/datasets/packaged/74_hiveformer_tasks_train
# valset=/home/zhouxian/git/datasets/packaged/74_hiveformer_tasks_val
# num_workers=2
# train_cache_size=0
# val_cache_size=0

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
task=put_umbrella_in_umbrella_stand


task=insert_onto_square_peg
# task=reach_and_drag
# task=push_button

train_iters=500000

batch_size_val=4
lr=1e-4

gripper_bounds_buffer=0.04
use_instruction=0
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

regress_position_offset=0
vis_ins_att_complex=0
vis_ins_att=0

new_rotation_loss=0
batch_size=16
ins_pos_emb=0
instruction_file=instructions_local.pkl
symmetric_rotation_loss=0
disc_rot=1
disc_rot_res=5.0
disc_rot_smooth=6.0
rotation_loss_coeff=1

batch_size=12
new_rotation_loss=1
disc_rot=0
rotation_loss_coeff=1.0

high_res=1
num_sampling_level=3

python train.py\
     --devices cuda:0\
     --instructions instructions_old/$instruction_file \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --train_cache_size $train_cache_size \
     --val_cache_size $val_cache_size \
     --train_iters $train_iters \
     --num_workers $num_workers \
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --high_res $high_res\
     --simplify $simplify\
     --rotation_loss_coeff $rotation_loss_coeff\
     --simplify_ins $simplify_ins\
     --ins_pos_emb $ins_pos_emb\
     --vis_ins_att $vis_ins_att\
     --vis_ins_att_complex $vis_ins_att_complex\
     --disc_rot $disc_rot\
     --disc_rot_res $disc_rot_res\
     --disc_rot_smooth $disc_rot_smooth\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --batch_size_val $batch_size_val \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --num_ghost_points_val $num_ghost_points_val\
     --max_episodes_per_taskvar $max_episodes_per_taskvar\
     --symmetric_rotation_loss $symmetric_rotation_loss\
     --new_rotation_loss $new_rotation_loss\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --embedding_dim $embedding_dim\
     --num_ghost_point_cross_attn_layers $n_layer\
     --num_query_cross_attn_layers $n_layer\
     --num_vis_ins_attn_layers $n_layer\
     --seed $seed\
     --lr $lr\
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-T$num_ghost_points-V$num_ghost_points_val-symrot$symmetric_rotation_loss-newrot$new_rotation_loss-gptie$gp_emb_tying-simp$simplify-B$batch_size-demo$max_episodes_per_taskvar-dim$embedding_dim-L$n_layer-lr$lr-seed$seed-simpins$simplify_ins-ins_pos_emb$ins_pos_emb-vis_ins_att$vis_ins_att-disc_rot$disc_rot-$disc_rot_res-$disc_rot_smooth-rotcoef$rotation_loss_coeff-ins$instruction_file

