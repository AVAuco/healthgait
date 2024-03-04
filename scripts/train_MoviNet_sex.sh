#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait


seed=27
epochs=10000
img_size=224
num_frames=40
batch_size=32
learning_rate=0.001
id_experiment='00'
wandb_project='prueba'

data_path='/opt2/data/datasets/UCA-GAIT/'
patients_info='/opt/data/jzafra/thesis/scripts/patients_measures.csv'
save_dir='/opt2/data/datasets/UCA-GAIT/healthgait/train_results/MoviNet/sex_classification'
partitions_path='/opt2/data/datasets/UCA-GAIT/code/partitions/sex_classification'

cd /opt2/data/datasets/UCA-GAIT/healthgait/code/train


target='Sex'
classes_names="woman man"

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

    python train_MoviNet_classification.py --partitions_file "$partitions_path/partition_$partition.json" \
    --data_path $data_path \
    --patients_info $patients_info \
    --num_frames $num_frames \
    --img_size $img_size \
    --batch_size $batch_size \
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
    --classes_names $classes_names \
    --target $target \
    --experiment_name "$experiment_name" \
    $opt_flow_param
}

for model_id in 'a0' 'a5'; do
    for data_type in 'silhouette' 'semantic_segmentation'; do
        for data_class in 'WoJ' 'WJ' 'both'; do
            for partition in {0..3}; do
                run_training $data_type "" $partition 0 $data_class $model_id &
                pid1=$!
                wait $pid1
            done

            python means_test.py --results_path "$experiment_name"

        done
    done
done



for model_id in 'a0' 'a5'; do
    for optical_flow_method in 'GMFLOW' 'TVL1'; do
        for data_class in 'WoJ' 'WJ' 'both'; do
            for partition in {0..3}; do
                run_training 'optical_flow' $optical_flow_method $partition 0 $data_class $model_id &
                pid1=$!
                wait $pid1
            done

            python means_test.py --results_path "$experiment_name"

        done
    done
done


