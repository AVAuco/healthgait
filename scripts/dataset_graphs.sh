#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait



python dataset_graphs.py --patients_measures '/opt2/data/datasets/UCA-GAIT/patients_measures.csv' \
    --save_path '/opt2/data/datasets/UCA-GAIT/code/visualization/graphs' \
    --videos_path '/opt2/data/datasets/UCA-GAIT/raw' \
    --gait_parameters '/opt2/data/datasets/UCA-GAIT/gait_parameters.csv'