#!/bin/bash

source /opt/data/jzafra/miniconda3/etc/profile.d/conda.sh
conda activate uca_gait


# Command line arguments

input_file=""
verbose=0
f_provided=0 

while getopts "c:" opt; do
  case ${opt} in
    c )
      input_file=$OPTARG
      f_provided=1
      ;;
    \? ) echo "Uso: cmd -c config"
         exit 1
      ;;
  esac
done

if [ $f_provided -eq 0 ]; then
    echo "La opción -c es obligatoria."
    echo "Uso: cmd -c config"
    exit 1
fi

gait_parameters=`jq -r '.gait_parameters' "$input_file"`
gait_parameters_estimation=`jq -r '.gait_parameters_estimation' "$input_file"`
patients_measures=`jq -r '.patients_measures' "$input_file"`
partitions_path=`jq -r '.sex_partitions_path' "$input_file"`

seed=`jq -r '.XGBoost.train_sex.seed' "$input_file"`
njobs=`jq -r '.XGBoost.train_sex.njobs' "$input_file"`
splits=`jq -r '.XGBoost.train_sex.splits' "$input_file"`

grid_search=`jq -r '.XGBoost.train_sex.hyperparameters_search_space' "$input_file"`

base_hyperparameters_output=`jq -r '.base_hyperparameters_output' "$input_file"`

base_hyperparameters_output='/opt2/data/datasets/UCA-GAIT/code/hyperparameters/xgboost/sex/'
base_evaluation_output='/opt2/data/datasets/UCA-GAIT/code/evaluation/xgboost/sex/'


n_decimals=3
target='Sex'
classes_names='woman man'

# Define las cadenas de características
fast_gait="Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
usual_gait="Step_UGS Stride_UGS Cadence_UGS Velocity_UGS"
circumference="WaistC HipC NeckC"

# Mapeo de combinaciones a nombres descriptivos
declare -A combinations=(
    ["$fast_gait"]="fast_gait"
    ["$usual_gait"]="usual_gait"
    ["$circumference"]="circumference"
    ["$fast_gait $circumference"]="fast_gait_circumference"
    ["$usual_gait $circumference"]="usual_gait_circumference"
    ["$fast_gait $usual_gait"]="fast_usual_gait"
    ["$fast_gait $usual_gait $circumference"]="fast_usual_gait_circumference"
)

run_evaluation() {
    features="$1"
    gait_parameters_path="$2"
    combination_name="${combinations["$features"]}"

    # Verifica la ruta de gait_parameters_path y ajusta el sufijo correspondiente
    if [[ "$gait_parameters_path" == *"$gait_parameters" ]]; then
        suffix="OptoGait/"
    else
        suffix="Estimated/"
    fi

    hyperparameters_output="${base_hyperparameters_output}${combination_name}/${suffix}"
    evaluation_output="${base_evaluation_output}${combination_name}/${suffix}"

    #echo "Ejecutando con características: $features"
    #echo "Guardando en: $hyperparameters_output y $evaluation_output"



    # python train_xgboost_classification.py --gait_parameters "$gait_parameters_path" \
    #                     --patients_measures "$patients_measures" \
    #                     --partitions_path "$partitions_path" \
    #                     --features "$features" \
    #                     --target "$target" \
    #                     --seed "$seed" \
    #                     --splits "$splits" \
    #                     --hyperparameters_path "$hyperparameters_output" \
    #                     --classes_names "$classes_names" \
    #                     --evaluation_path "$evaluation_output" \
    #                     --grid_search "$grid_search" \
    #                     --njobs "$njobs" \
    #                     --n_decimals "$n_decimals"
 }

# Acceder al directorio con los scripts de entrenamiento
cd ../code/train/

for features in "${!combinations[@]}"; do
    for gait_parameters_path in "$gait_parameters" "$gait_parameters_estimation"; do
        run_evaluation "$features" "$gait_parameters_path"
    done
done
