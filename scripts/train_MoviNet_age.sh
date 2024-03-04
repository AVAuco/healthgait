#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

seed=27
epochs=10000
img_size=224
num_frames=40
batch_size=32
learning_rate=0.0001
units=512

data_path='/opt2/data/datasets/UCA-GAIT/'
patients_info='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'

id_experiment='00'

wandb_project='age_regression'
target='Age'
save_dir='/opt2/data/datasets/UCA-GAIT/train_results/age_regression/'
#partitions_path='/opt2/data/datasets/UCA-GAIT/code/partitions/weight_regression'
partitions_path='/opt2/data/datasets/UCA-GAIT/code/partitions/age_regression'


run_training() {
    data_type=$1
    optical_flow_method=$2
    partition=$3
    device=$4
    data_class=$5
    model_id=$6
    experiment_name="$id_experiment: Data_type $data_type $optical_flow_method, Data_Class $data_class, Num_Frames $num_frames, Model_ID $model_id, SEED $seed"

    if [ "$data_type" = "optical_flow" ]; then
        opt_flow_param="--optical_flow_method $optical_flow_method"
    else
        opt_flow_param=""
    fi

    python train_MoviNet_regression.py --partitions_file "$partitions_path/partition_$partition.json" \
    --data_path $data_path \
    --patients_info $patients_info \
    --num_frames $num_frames \
    --img_size $img_size \
    --batch_size $batch_size \
    --units $units \
    --model_id $model_id \
    --learning_rate $learning_rate \
    --id_partition $partition \
    --save_dir $save_dir \
    --epochs $epochs \
    --device $device \
    --data_class $data_class \
    --data_type $data_type \
    --id_experiment $id_experiment \
    --seed $seed \
    --wandb_project $wandb_project \
    --target $target \
    --experiment_name "$experiment_name" \
    $opt_flow_param
}


# for data_type in 'silhouette' 'semantic_segmentation'; do
#     for partition in {0..3}; do
#         run_training $data_type "" $partition 1 'both' 'a5' &
#         pid1=$!
#         wait $pid1
#     done

#     #python means_test.py --results_path "$experiment_name"

# done

for data_type in 'semantic_segmentation'; do

    run_training $data_type "" 3 1 'both' 'a5' &
    pid1=$!
    wait $pid1

done


for optical_flow_method in 'GMFLOW' 'TVL1'; do

    for partition in {0..3}; do
        run_training 'optical_flow' $optical_flow_method $partition 1 'both' 'a5' &
        pid1=$!
        wait $pid1
    done

    #python means_test.py --results_path "$experiment_name"

done



