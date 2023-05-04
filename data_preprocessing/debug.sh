
root=/home/zhouxian/git
data_dir=$root/datasets/raw
task_file=tasks/stack_wine.csv

image_size="256,256"

cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/74_hiveformer_tasks_debug \
    --tasks=wipe_desk \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=10 \
    --variations=-1 \
    --offset=0 \
    --processes=1