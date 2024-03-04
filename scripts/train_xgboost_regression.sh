#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

gait_parameters='/opt2/data/datasets/UCA-GAIT/gait_parameters_estimation.csv'
patients_measures='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'
partitions_path='/opt2/data/datasets/UCA-GAIT/code/partitions/age_regression'
hyperparameters_path='/opt2/data/datasets/UCA-GAIT/code/hyperparameters/xgboost/age/usualfast_gait_estimated'
evaluation_path='/opt2/data/datasets/UCA-GAIT/code/evaluation/xgboost/age/usual_fastgait_estimated'
grid_search='/opt2/data/datasets/UCA-GAIT/code/train/xgboost/grid_search.json'
seed=27
splits=3
njobs=10
n_decimals=3

# Step_FGS Stride_FGS Cadence_FGS Velocity_FGS
features="Step_UGS Stride_UGS Cadence_UGS Velocity_UGS Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
target='Age'

python train_xgboost_regression.py --gait_parameters $gait_parameters \
                        --patients_measures $patients_measures \
                        --partitions_path $partitions_path \
                        --features $features \
                        --target $target \
                        --seed $seed \
                        --splits $splits \
                        --hyperparameters_path $hyperparameters_path \
                        --evaluation_path $evaluation_path \
                        --grid_search $grid_search \
                        --njobs $njobs \
                        --n_decimals $n_decimals