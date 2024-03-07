#!/bin/bash

# Command line arguments

patient_file=""
patient_provided=0 

output_path=""
output_provided=0

validation_ratio=0.10
seed=27
k_fold=4

while getopts "p:o:v:s:k:" opt; do
  case ${opt} in
    p )
      patient_file=$OPTARG
      patient_provided=1
      ;;
    o )
      output_path=$OPTARG
      output_provided=1
      ;;
    v )
      validation_ratio=$OPTARG
      ;;
    s )
        seed=$OPTARG
        ;;
    k )
        k_fold=$OPTARG
        ;;
    \? ) echo "Usage: cmd -p <patient_measures_file> -o <output_path> -v [validation_ratio] -s [seed] -k [k_fold]"
         exit 1
      ;;
  esac
done

if [ $patient_provided -eq 0 ] || [ $output_provided -eq 0 ]; then
    echo "Error: Missing arguments."
    echo "Usage: cmd -p <patient_measures_file> -o <output_path> -v [validation_ratio] -s [seed] -k [k_fold]"
    exit 1
fi

cd ../code/create_partitions/

variables="Age Sex"

#SEX PARTITIONS
python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patient_file \
    --output_path "$output_path/Sex" \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose


# WEIGHT PARTITIONS

variables="Sex BMI"

python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patient_file \
    --output_path "$output_path/Weight" \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose


# AGE REGRESSION PARTITIONS

variables="Age Sex"

python create_partitions.py --validation_ratio $validation_ratio \
    --patients_measures $patient_file \
    --output_path "$output_path/Age" \
    --k_fold $k_fold \
    --seed $seed \
    --variables $variables \
    --verbose