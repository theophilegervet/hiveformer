# Data Generation

## 1 - HiveFormer Data Generation
```
root=/home/zhouxian/git
data_dir=$root/datasets/raw
output_dir=$root/datasets/packaged
train_dir=74_hiveformer_tasks_train
val_dir=74_hiveformer_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/stack_wine.csv

nohup sudo X &
export DISPLAY=:0.0
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=1

python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=5
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=1 \
            --tasks=$task
    done
done
```

## 1 - PerAct Data Generation
```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/raw
output_dir=$root/datasets/packaged
train_dir=18_peract_tasks_train
val_dir=18_peract_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/18_peract_tasks.csv
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=5
    
python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=5
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=60 \
            --tasks=$task
    done
done
```

## 3 - Preprocess Instructions for Both Datasets
```
root=/home/theophile_gervet_gmail_com
cd $root/hiveformer

task_file=tasks/82_all_tasks.csv
python -m data_preprocessing.preprocess_instructions \
    --tasks $(cat $task_file | tr '\n' ' ') \
    --output instructions.pkl \
    --variations {0..199} \
    --annotations data_preprocessing/annotations.json
```

## 4 - Compute Workspace Bounds for Both Datasets
```
root=/home/theophile_gervet_gmail_com
cd $root/hiveformer

output_dir=$root/datasets/packaged
train_dir=74_hiveformer_tasks_train
task_file=tasks/74_hiveformer_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_dir \
    --out_file 74_hiveformer_tasks_location_bounds.json \
    --tasks $(cat $task_file | tr '\n' ' ')

output_dir=$root/datasets/packaged
train_dir=18_peract_tasks_train
task_file=tasks/18_peract_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_dir \
    --out_file 18_peract_tasks_location_bounds.json \
    --variations {0..199} \
    --tasks $(cat $task_file | tr '\n' ' ')
```


python -m data_preprocessing.compute_workspace_bounds \
    --dataset /home/zhouxian/git/datasets/packaged/diffusion_trajectories_train \
    --out_file diffusion_location_bounds.json \
    --instructions instructions_new.pkl \
    --tasks close_door