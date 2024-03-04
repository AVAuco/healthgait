#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

img_size=224
batch_size=32
learning_rate=0.0001
units=512
data_path='/opt2/data/datasets/UCA-GAIT/'
patients_info='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'
seed=27
n_decimals=3

#model_id='a5'
#data_class='both'
num_frames=40
device=1
id_experiment='00'


run_evaluation() {
    data_type=$1
    optical_flow_method=$2
    partition=$3
    device=$4
    data_class=$5
    model_id=$6
    save_dir=$7

    if [ "$data_type" = "optical_flow" ]; then
        opt_flow_param="--optical_flow_method $optical_flow_method"
        checkpoint="/opt2/data/datasets/UCA-GAIT/train_results/age_regression/$id_experiment: Data_type $data_type $optical_flow_method, Data_Class $data_class, Num_Frames $num_frames, Model_ID $model_id, SEED $seed/Partition $partition/checkpoint/checkpoint"
    else
        opt_flow_param=""
        checkpoint="/opt2/data/datasets/UCA-GAIT/train_results/age_regression/$id_experiment: Data_type $data_type , Data_Class $data_class, Num_Frames $num_frames, Model_ID $model_id, SEED $seed/Partition $partition/checkpoint/checkpoint"
    fi

    python evaluate_MoviNet_regression.py --device 0 \
        --img_size $img_size \
        --data_path $data_path \
        --data_type $data_type \
        --num_frames $num_frames \
        --partitions_file "/opt2/data/datasets/UCA-GAIT/code/partitions/age_regression/partition_$partition.json" \
        --patients_info $patients_info \
        --batch_size $batch_size \
        --data_class $data_class \
        --seed $seed \
        --target 'Age' \
        --save_dir $save_dir \
        --n_decimals $n_decimals \
        --learning_rate $learning_rate \
        --checkpoint "$checkpoint" \
        --units $units \
        --model_id $model_id \
        $opt_flow_param
}


for data_type in 'silhouette'; do
    save_dir="/opt2/data/datasets/UCA-GAIT/code/evaluation/MoviNet/evaluations/age_regression/$data_type"
    for partition in {0..3}; do
        run_evaluation $data_type "" $partition 0 'both' 'a5' $save_dir &
        pid1=$!
        wait $pid1
    done
   python get_means.py --results_path "/opt2/data/datasets/UCA-GAIT/code/evaluation/MoviNet/evaluations/age_regression/$data_type" --n_decimals 3
done

# for optical_flow_method in 'GMFLOW' 'TVL1'; do
#     save_dir="/opt2/data/datasets/UCA-GAIT/code/evaluation/MoviNet/evaluations/age_regression/$optical_flow_method"
#     for partition in {0..3}; do
#         run_evaluation 'optical_flow' $optical_flow_method $partition 0 'both' 'a5' $save_dir &
#         pid1=$!
#         wait $pid1
#     done
#     python get_means.py --results_path "/opt2/data/datasets/UCA-GAIT/code/evaluation/MoviNet/evaluations/age_regression/$optical_flow_method" --n_decimals 3
# done


    

