# To Google Cloud through local

#source_prefix=home/tgervet/hiveformer/train_logs
#target_prefix=home/theophile_gervet_gmail_com/hiveformer
#exp_src=03_13_peract_setting
#exp_tgt=03_13_peract_setting
#ckpt=best.pth
#
## Get Tensorboard from source
##sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/*/events.out*" .
#
## Get checkpoints from source
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/*/$ckpt" .
#
## Send all to target
#scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/


# To Xian through local

source_prefix=private/home/theop123/hiveformer/train_logs
#target_prefix=home/zhouxian/git/hiveformer_theo
target_prefix=home/sirdome/katefgroup/hiveformer2
exp_src=hiveformer_10_episodes
exp_tgt=hiveformer_10_episodes
ckpt=best.pth

# Get Tensorboard from source
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/*/events.out*" .

# Get checkpoints from source
rsync -RL "devfair:/$source_prefix/$exp_src/*/$ckpt" .

# Send all to target
#sshpass -p $LAB_PW scp -r $source_prefix/$exp_tgt lab:/$target_prefix/
sshpass -p $LAB2_PW scp -r $source_prefix/$exp_tgt lab2:/$target_prefix/


# To Xian directly from Matrix

#source_prefix=home/tgervet/hiveformer/train_logs
#target_prefix=/home/sirdome/katefgroup/hiveformer
#exp_src=03_24_hiveformer_setting
#exp_tgt=03_24_hiveformer_setting
#ckpt=best.pth
#
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/*/$ckpt" .


# Uncleaned HiveFormer

#source_prefix=home/tgervet/hiveformer/train_logs
#source_prefix=home/tgervet/hiveformer/vln-robot/alhistory/xp
#exp_src=uncleaned_code_02_15
#exp_tgt=uncleaned_code_02_15
#ckpt=checkpoints/epoch=0-step=90000.ckpt
#
## Get Tensorboard from source
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/lightning_logs/*/events.out*" .
#
## Get checkpoints from source
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/lightning_logs/*/$ckpt" .
#
## Send all to target
#scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/
