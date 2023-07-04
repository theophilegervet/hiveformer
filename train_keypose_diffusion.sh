CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    main_keypose.py --tasks reach_target \
    --n_gpus 1 --num_workers 4 \
    --dataset /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train/ \
    --valset /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val/ \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --gripper_loc_bounds tasks/74_hiveformer_tasks_location_bounds.json \
    --use_instruction 0 \
    --model regression \
    --exp_log_dir reach_target_regression_keypose_restart_n \
    --num_query_cross_attn_layers 2 \
    --val_freq 1000 \
    --batch_size 24 --batch_size_val 100 --lr 1e-4 \
    --cache_size 100 \
    --cache_size_val 100 \
    --attn_rounds 2 --feat_scales_to_use 2 \
    --action_dim 7 \
    # --checkpoint /home/ngkanats/analogical_manipulation/train_logs/reach_target_diffusion_keypose_restart_multi/run/last.pth
    # --checkpoint /home/ngkanats/analogical_manipulation/train_logs/reach_target_no_lang_regression_keypose_abs2/run/last.pth
    # --checkpoint /home/ngkanats/analogical_manipulation/train_logs/take_frame_off_hanger_no_lang_diffusion_keypose2/run/last.pth
