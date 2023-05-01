# valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val/
# valset=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_train
# valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val_fail/
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
valset=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val

task=reach_target
task=push_button
# task=slide_block_to_target
# task=pick_up_cup
# task=take_umbrella_out_of_umbrella_stand
task=pick_and_lift
task=put_knife_on_chopping_board
# task=take_money_out_safe
task=put_money_in_safe
# task=stack_wine
task=insert_onto_square_peg
# task=reach_and_drag

gripper_bounds_buffer=0.04
weight_tying=1

gp_emb_tying=1
simplify=1
num_sampling_level=3
num_ghost_points_val=20000
embedding_dim=60
n_layer=2
randomize_vp=0
use_instruction=1

simplify_ins=0
ins_pos_emb=1
vis_ins_att=1
vis_ins_att_complex=0
regress_position_offset=0

ckpt=/home/zhouxian/git/hiveformer/train_logs/04_16_multitask_cont/10_tasks-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B32-demo100-dim60-L2-lr1e-4-seed0-simpins0-ins_pos_emb1-vis_ins_att1-vis_ins_att_complex0_version164462/model.step=220000-value=0.00000.pth

# simplify_ins=0
# ins_pos_emb=0
# vis_ins_att=0
# vis_ins_att_complex=0
# regress_position_offset=0
# ckpt=/home/zhouxian/git/hiveformer/train_logs/04_10_multitask_revert/10_tasks-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo100-dim60-L2-lr1e-4-seed0-simpins0_version163486/model.step=280000-value=0.00000.pth

simplify_ins=0
ins_pos_emb=0
vis_ins_att=0
vis_ins_att_complex=0
regress_position_offset=0
ckpt=/home/zhouxian/git/hiveformer/train_logs/04_05_multitask/10_tasks-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo100-dim60-L2-lr1e-4-seed0_version162732/model.step=200000-value=0.00000.pth

vis_ins_att=1
use_instruction=0
ckpt=/home/zhouxian/git/hiveformer/train_logs/04_28_singletask/insert_onto_square_peg-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B8-demo100-dim60-L2-lr1e-4-seed0-simpins0-ins_pos_emb0-vis_ins_att1-vis_ins_att_complex0-insinstructions_local.pkl_version167170/model.step=250000-value=0.00000.pth

instruction_file=instructions_local.pkl
python eval.py\
     --instructions instructions_old/$instruction_file \
     --tasks $task\
     --checkpoint $ckpt \
     --data_dir $valset\
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --simplify $simplify\
     --simplify_ins $simplify_ins\
     --ins_pos_emb $ins_pos_emb\
     --vis_ins_att $vis_ins_att\
     --vis_ins_att_complex $vis_ins_att_complex\
     --image_size 256,256\
     --offline 0\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --num_ghost_points_val $num_ghost_points_val\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --embedding_dim $embedding_dim\
     --num_ghost_point_cross_attn_layers $n_layer\
     --num_query_cross_attn_layers $n_layer\
     --run_log_dir $task-ONLINE\
     --randomize_vp $randomize_vp\
     # --max_episodes 20
