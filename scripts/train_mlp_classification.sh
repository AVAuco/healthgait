#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait



gait_parameters='/opt2/data/datasets/UCA-GAIT/gait_parameters_estimation.csv'
patients_measures='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'
#partitions_path='/opt/data/jzafra/thesis/technical_validation/partitions_xgboost/sex_classification'
partitions_path='/opt2/data/datasets/UCA-GAIT/code/partitions/sex_classification'
hyperparameters_path='/opt2/data/datasets/UCA-GAIT/code/hyperparameters/MLP/sex/usualfast_gait_circumf_estimated'
evaluation_path='/opt2/data/datasets/UCA-GAIT/code/evaluation/MLP/sex/usualfast_gait_circumf_estimated'
seed=27 
n_decimals=3

target='Sex'

#features="Step_FGS Stride_FGS Cadence_FGS MonoSP_FGS BiSP_FGS Velocity_FGS WaistC HipC NeckC Step_UGS Stride_UGS Cadence_UGS MonoSP_UGS BiSP_UGS Velocity_UGS"
features="Step_UGS Stride_UGS Cadence_UGS Velocity_UGS Step_FGS Stride_FGS Cadence_FGS Velocity_FGS WaistC HipC NeckC"
#features="WaistC HipC NeckC"

classes_names='woman man'

python train_mlp_classification.py --gait_parameters $gait_parameters \
                        --patients_measures $patients_measures \
                        --partitions_path $partitions_path \
                        --features $features \
                        --target $target \
                        --seed $seed \
                        --hyperparameters_path $hyperparameters_path \
                        --evaluation_path $evaluation_path \
                        --n_decimals $n_decimals \
                        --classes_names $classes_names