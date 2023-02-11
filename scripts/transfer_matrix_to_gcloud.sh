source_prefix=home/tgervet/hiveformer/train_logs
target_prefix=home/theophile_gervet_gmail_com/hiveformer
exp_src=reproduce_hiveformer
exp_tgt=reproduce_hiveformer
ckpt=best.pth

# Get Tensorboard from source
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/*/events.out*" .

# Get checkpoints from source
sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/*/$ckpt" .

# Send all to target
scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/