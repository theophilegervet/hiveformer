main_dir=reach_target_debug_reg2_w1000

dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

task=reach_target


batch_size_val=4
batch_size=16
lr=1e-4

gripper_bounds_buffer=0.04
use_instruction=0
weight_tying=1
max_episodes_per_taskvar=100
symmetric_rotation_loss=0
num_ghost_points=1000
num_ghost_points_val=10000

gp_emb_tying=1
simplify=1
num_sampling_level=3
regress_position_offset=0
seed=0
embedding_dim=24
embedding_dim=60


CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    main_act3d.py \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --instructions /home/tgervet/hiveformer/instructions.pkl \
     --gripper_loc_bounds tasks/74_hiveformer_tasks_location_bounds.json \
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --batch_size_val $batch_size_val \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --num_ghost_points_val $num_ghost_points_val\
     --symmetric_rotation_loss $symmetric_rotation_loss\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --embedding_dim $embedding_dim\
     --seed $seed\
     --lr $lr\
     --val_freq 1000\
     --position_loss_coeff 1 \
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-T$num_ghost_points-V$num_ghost_points_val-symrot$symmetric_rotation_loss-gptie$gp_emb_tying-simp$simplify-B$batch_size-demo$max_episodes_per_taskvar-dim$embedding_dim-lr$lr-seed$seed\
    #  --checkpoint /home/ngkanats/analogical_manipulation/train_logs/reach_target_debug_reg/reach_target-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo100-dim60-lr1e-4-seed0/last.pth

