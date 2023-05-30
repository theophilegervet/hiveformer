rsync -avzh  --info=progress2 --exclude-from='/home/zhouxian/git/hiveformer/xian_misc/rsync-exclude-file.txt' /home/zhouxian/git/hiveformer xianz1@matrix.ml.cmu.edu:/home/xianz1/git/
rsync -avzh --info=progress2  --exclude-from='/home/zhouxian/git/hiveformer/xian_misc/rsync-exclude-pth.txt' xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/diffuse*  /home/zhouxian/git/hiveformer/train_logs/
rsync -avzh --info=progress2  xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/instruction*  /home/zhouxian/git/hiveformer/

rsync -avzh --info=progress2 xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/diffuse_05_28/wipe_desk-B24-lr1e-4-DI1-50-PL0 /home/zhouxian/git/hiveformer/train_logs/diffuse_05_28/

rsync -avzh --info=progress2 xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-50_version174697/model.step=415000-value=0.00000.pth /home/zhouxian/git/hiveformer/train_logs/diffuse_05_25/close_door-B24-lr1e-4-DI1-50_version174697/

rsync -avzh --info=progress2 xianz1@matrix.ml.cmu.edu:/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/close_door+0 /home/zhouxian/git/datasets/packaged/diffusion_trajectories_train/

rsync -avzh --info=progress2 --exclude *.png sirdome@128.2.176.111:/home/sirdome/katefgroup/datasets/raw/diffusion_trajectories_val /home/zhouxian/git/datasets/raw/

squeue -o "%u %c %m %b %P %M %N" |grep 2-25
squeue -o "%u %c %m %b %P %M %N" |grep 2-29
squeue -o "%u %c %m %b %P %M %N" |grep 1-22
squeue -o "%u %c %m %b %P %M %N" |grep 0-36
squeue -o "%u %c %m %b %P %M %N" |grep 1-24
squeue -o "%u %c %m %b %P %M %N" |grep 1-14
squeue -o "%u %c %m %b %P %M %N" |grep 0-16
squeue -o "%u %c %m %b %P %M %N" |grep 0-18
squeue -o "%u %c %m %b %P %M %N" |grep 0-22
# A100
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=62g  --nodelist=matrix-2-25 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=62g  --nodelist=matrix-2-29 --pty $SHELL 
# 2080ti
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-1-22 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=128g  --nodelist=matrix-1-22 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-36 --pty $SHELL 
# V100
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c20 --mem=100g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=240g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:2 -c20 --mem=128g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:3 -c30 --mem=180g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=240g  --nodelist=matrix-1-14 --pty $SHELL 
# Titan X
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-0-16 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-18 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-22 --pty $SHELL 