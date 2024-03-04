#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait

pose_path='/opt2/data/datasets/UCA-GAIT/pose'
sensors_path='/opt2/data/datasets/UCA-GAIT/sensors_bboxes'
segmentation_path='/opt2/data/datasets/UCA-GAIT/semantic_segmentation'
scale=4.2
fps=29.97
csv_output='/opt2/data/datasets/UCA-GAIT/gait_parameters_estimation.csv'

python gait_parameters_estimation.py --pose_path $pose_path \
    --sensors_path $sensors_path \
    --segmentation_path $segmentation_path \
    --scale $scale \
    --csv_output $csv_output \
    --fps $fps



