python train_diffusion.py --tasks unplug_charger \
    --dataset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/ \
    --valset /projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/ \
    --instructions /home/tgervet/hiveformer/instructions.pkl \
    --use_instruction 1 \
    --exp_log_dir test \
    --batch_size 84 --batch_size_val 42 --lr 3e-4


# /projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train