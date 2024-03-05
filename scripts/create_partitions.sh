#!/bin/bash


source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

validation_ratio=0.10
patients_measures='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'
seed=27
k_fold=4


# variables="Age Sex"

SEX PARTITIONS
python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patients_measures \
    --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/Sex' \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose


# WEIGHT PARTITIONS

# variables="Sex BMI"

python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patients_measures \
    --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/Weight' \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose


# AGE REGRESSION PARTITIONS

variables="Age Sex"

python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patients_measures \
    --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/Age' \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose