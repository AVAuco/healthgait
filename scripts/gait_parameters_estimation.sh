#!/bin/bash

# Command line arguments

pose_file=""
pose_provided=0 

sensors_path=""
sensors_provided=0

semantic_segmentation_path=""
segmentation_provided=0

output_path=""
output_provided=0

scale=4.2
fps=29.97

while getopts "p:s:e:o:k:f:" opt; do
  case ${opt} in
    p )
      pose_file=$OPTARG
      pose_provided=1
      ;;
    s )
        sensors_path=$OPTARG
        sensors_provided=1
        ;;
    e )
        semantic_segmentation_path=$OPTARG
        segmentation_provided=1
        ;;
    o )
        output_path=$OPTARG
        output_provided=1
        ;;
    k )
        scale=$OPTARG
        ;;
    f )
        fps=$OPTARG
        ;;
    \? ) echo "Usage: cmd -p <patient_measures_file> -s <sensor_bboxes> -e <semantic_segmentation_path> -o <output_csv_path> -k [scale] -f [fps]"
         exit 1
      ;;
  esac
done

if [ $pose_provided -eq 0 ] || [ $sensors_provided -eq 0 ] || [ $segmentation_provided -eq 0 ] || [ $output_provided -eq 0 ]; then
    echo "Error: Missing arguments."
    echo "Usage: cmd -p <patient_measures_file> -s <sensor_bboxes> -e <semantic_segmentation_path> -o <output_csv_path> -k [scale] -f [fps]"
    exit 1
fi

cd ../code/gait_estimation/


#pose_path='/opt2/data/datasets/UCA-GAIT/pose'
#sensors_path='/opt2/data/datasets/UCA-GAIT/sensors_bboxes'
#segmentation_path='/opt2/data/datasets/UCA-GAIT/semantic_segmentation'

#csv_output='/opt2/data/datasets/UCA-GAIT/gait_parameters_estimation.csv'

python gait_parameters_estimation.py --pose_path $pose_path \
    --sensors_path $sensors_path \
    --segmentation_path $segmentation_path \
    --scale $scale \
    --csv_output $output_path \
    --fps $fps



