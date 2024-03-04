#!/bin/bash


source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

validation_ratio=0.10
patients_measures='/opt2/data/datasets/UCA-GAIT/patients_measures.csv'
seed=27
k_fold=4


# variables="Age Sex"

# SEX CLASSIFICATION PARTITIONS
# python create_partitions.py --validation_ratio $validation_ratio \
#     --patients_measures $patients_measures \
#     --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/sex' \
#     --k_fold $k_fold \
#     --seed $seed \
#     --variables $variables \
#     --verbose

#ACTIVITY LEVEL CLASSIFICATION PARTITIONS

# YOUNG PARTITIONS

# variables="Age Sex PA_level"

# python create_partitions.py --validation_ratio $validation_ratio \
#     --patients_measures $patients_measures \
#     --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/activity_classification/young' \
#     --k_fold $k_fold \
#     --seed $seed \
#     --age_range 'young' \
#     --variables $variables\
#     --verbose

# ADULT PARTITIONS

# python create_partitions.py --validation_ratio $validation_ratio \
#     --patients_measures $patients_measures \
#     --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/activity_classification/adult' \
#     --k_fold $k_fold \
#     --seed $seed \
#     --age_range 'adult' \
#     --variables $variables\
#     --verbose

# OLD PARTITIONS

# python create_partitions.py --validation_ratio $validation_ratio \
#     --patients_measures $patients_measures \
#     --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/activity_classification/old' \
#     --k_fold $k_fold \
#     --seed $seed \
#     --age_range 'old' \
#     --variables $variables\
#     --verbose


# WEIGHT REGRESSION PARTITIONS

# variables="Sex BMI"

# python create_partitions.py --validation_ratio $validation_ratio \
#     --patients_measures $patients_measures \
#     --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/weight' \
#     --k_fold $k_fold \
#     --seed $seed \
#     --variables $variables \
#     --verbose


# AGE REGRESSION PARTITIONS

variables="Age Sex"

python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patients_measures \
    --output_path '/opt2/data/datasets/UCA-GAIT/code/partitions/age' \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose