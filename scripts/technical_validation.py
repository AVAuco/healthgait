import subprocess
import argparse
import json
import os

def parse_args():

    parser = argparse.ArgumentParser(description="Technical validation of the HealthGait dataset.")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--device", type=int, default=0, help="Device to run the experiments")
    parser.add_argument("--targets", nargs='+', required=True, help="list with the targets to estimate")
    parser.add_argument('--methods', nargs='+', required=True, help='list with the clases names to classifier')

    return parser.parse_args()


def load_config(config_path):
    
    with open(config_path, "r") as f:

        config = json.load(f)

    return config


def base_command_MoviNet(mode, partitions_path, data_path, patients_measures, 
                         save_dir, device, target, model_id, data_type, 
                         seed, num_frames, img_size, batch_size, learning_rate, 
                         data_class, partition, id_experiment, epochs=None,
                         wandb_project=None, classes_names=None, units=None, 
                         optical_flow_method=None, n_decimals=None):
    
    experiment_name = f"{id_experiment}: Data_type {data_type}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"

    
    base_command = f"python {mode}_MoviNet_{'classification' if target == 'Sex' else 'regression'}.py --partitions_file {partitions_path}/partition_{partition}.json " \
                    f"--data_path {data_path} " \
                    f"--patients_info {patients_measures} " \
                    f"--num_frames {num_frames} " \
                    f"--img_size {img_size} " \
                    f"--batch_size {batch_size} " \
                    f"--model_id {model_id} " \
                    f"--learning_rate {learning_rate} " \
                    f"--device {device} " \
                    f"--data_class {data_class} " \
                    f"--data_type {data_type} " \
                    f"--seed {seed} " \
                    f"--target {target} " \
                    f"--id_partition {partition} "
    
    if mode == 'train':

        save_dir = os.path.join(save_dir, "MoviNet")

        base_command += f"--epochs {epochs} " \
                        f"--id_experiment {id_experiment} " \
                        f"--wandb_project {wandb_project} " \
                        f"--experiment_name \"{experiment_name}\" " \
                        f"--save_dir \"{save_dir}\" "

    else:

        checkpoint = os.path.join(save_dir, "MoviNet", experiment_name, f"Partition {partition}", "checkpoint", "checkpoint")

        save_dir = os.path.join(save_dir, "MoviNet", experiment_name, f"Partition {partition}", "evaluation")

        base_command += f"--checkpoint \"{checkpoint}\" " \
                        f"--n_decimals {n_decimals} " \
                        f"--save_dir \"{save_dir}\" "
                        
    # For classification tasks, classes_names must be provided
    if target == 'Sex':
        base_command += f" --classes_names {classes_names} "
    # For regression tasks, units must be provided
    else:
        base_command += f" --units {units} "

    if optical_flow_method:
        base_command += f" --optical_flow_method {optical_flow_method}"

    return base_command


def base_command_XGBoost_or_MLP(gait_parameters_path, patients_measures, partitions_path,
                                features, target, seed, save_dir, n_decimals, method,
                                combination_name, suffix, splits=None, grid_search=None, 
                                njobs=None, classes_names=None):
    
    
    hyperparameters_output = f"{save_dir}/{method}/{target}/hyperparameters/{combination_name}/{suffix}"
    evaluation_output = f"{save_dir}/{method}/{target}/evaluation/{combination_name}/{suffix}"

    base_command = f"python train_{method}_{'classification' if target == 'Sex' else 'regression'}.py " \
                    f"--gait_parameters {gait_parameters_path} " \
                    f"--patients_measures {patients_measures} " \
                    f"--partitions_path {partitions_path} " \
                    f"--features {features} " \
                    f"--target {target} " \
                    f"--seed {seed} " \
                    f"--n_decimals {n_decimals} " \
                    f"--hyperparameters_path {hyperparameters_output} " \
                    f"--evaluation_path {evaluation_output} "
    
    if target == 'Sex':
        base_command += f"--classes_names {classes_names} "

    if method == "xgboost":
        base_command += f"--splits {splits} " \
                        f"--grid_search {grid_search} " \
                        f"--njobs {njobs}"
    
    return base_command

def run_command(command):

    subprocess.run(command, shell=True)


def run_experiments(config, method, target, device = 0):

    data_path = config["data_path"]
    save_dir = config["save_dir"]
    partitions_path = os.path.join(config["partitions_path"], target)
    patients_measures = config["patients_measures"]

    # Classification and regression movinet experiments 
    if method == "movinet":

        common_parameters = {
            "partitions_path": partitions_path,
            "data_path": data_path,
            "patients_measures": patients_measures,
            "device": device,
            "target": target,
            "save_dir": save_dir,
            "seed": config[method][f"train_{target}"]["seed"],
            "img_size": config[method][f"train_{target}"]["img_size"],
            "num_frames": config[method][f"train_{target}"]["num_frames"],
            "batch_size": config[method][f"train_{target}"]["batch_size"],
            "learning_rate": config[method][f"train_{target}"]["learning_rate"],
        }

        for model_id in ['a0', 'a5']:
            for data_type in ['silhouette', 'semantic_segmentation', 'optical_flow']:
                for data_class in ['WoJ', 'WJ', 'both']:
                    for partition in range(4):

                        print(f"Running MoviNet with target {target}, model_id {model_id}, data_type {data_type}, data_class {data_class}, partition {partition}...")

                        # Adjust command based on target and data type
                        if target == "Sex":
                            classes_names = "woman man"
                        else:
                            units = config[method][f"train_{target}"]["units"]
                            
                        optical_flow_methods = ['GMFLOW', 'TVL1'] if data_type == 'optical_flow' else [None]
                        
                        for optical_flow_method in optical_flow_methods:
                            command = base_command_MoviNet(
                                mode = "train",
                                epochs = config[method][f"train_{target}"]["epochs"],
                                id_experiment = config[method][f"train_{target}"]["id_experiment"],
                                wandb_project = config[method][f"train_{target}"]["wandb_project"],
                                classes_names=classes_names if target == 'Sex' else None,
                                units=units if target != 'Sex' else None,
                                model_id=model_id, 
                                data_type=data_type if optical_flow_method is None else f"{data_type} {optical_flow_method}",
                                partition=partition, 
                                optical_flow_method=optical_flow_method, data_class=data_class, **common_parameters
                            )

                            run_command(command)

                            command = base_command_MoviNet(
                                mode = "evaluate",
                                id_experiment = config[method][f"train_{target}"]["id_experiment"],
                                n_decimals = 3,
                                classes_names=classes_names if target == 'Sex' else None,
                                units=units if target != 'Sex' else None,
                                model_id=model_id, 
                                data_type=data_type if optical_flow_method is None else f"{data_type} {optical_flow_method}",
                                partition=partition, 
                                optical_flow_method=optical_flow_method, data_class=data_class, **common_parameters
                            )

                            run_command(command)

                            print("MoviNet finished")
                    
                    run_command(command)


                        
    # XGBoost and MLP experiments
    elif method == "xgboost" or method == "mlp":

        fast_gait = "Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
        usual_gait = "Step_UGS Stride_UGS Cadence_UGS Velocity_UGS"
        circumference = "WaistC HipC NeckC"

        combinations = {
            fast_gait: "fast_gait",
            usual_gait: "usual_gait",
            f"{fast_gait} {circumference}": "fast_gait_circumference",
            f"{usual_gait} {circumference}": "usual_gait_circumference",
            f"{fast_gait} {usual_gait}": "fast_usual_gait",
            f"{fast_gait} {usual_gait} {circumference}": "fast_usual_gait_circumference"
        }

        common_parameters = {
            "partitions_path": partitions_path,
            "patients_measures": patients_measures,
            "target": target,
            "save_dir": save_dir,
            "n_decimals": 3,
            "method": method,
            "seed": config[method][f"train_{target}"]["seed"]
        }

        if target == "Sex":
            classes_names = "woman man"
        else:
            classes_names = None


        for features, combination_name in combinations.items():
            for gait_parameters_path in [config['gait_parameters'], config['gait_parameters_estimation']]:

                if gait_parameters_path == config['gait_parameters']:
                    suffix = "OptoGait/"
                else:
                    suffix = "Estimated/"

                print(f"Running {method} with {combination_name} features...")

                command = base_command_XGBoost_or_MLP(classes_names=classes_names,
                                            splits=config["xgboost"][f"train_{target}"]["splits"] if method == "xgboost" else None,
                                            grid_search=config['xgboost'][f"train_{target}"]['hyperparameters_search_space'] if method == "xgboost" else None,
                                            njobs=config["xgboost"][f"train_{target}"]["njobs"] if method == "xgboost" else None, 
                                            suffix=suffix, features=features, combination_name=combination_name,
                                            gait_parameters_path=gait_parameters_path, **common_parameters)
                
                run_command(command)

                print(f"{method} finished")

        # Run XGBoost or MLP with circumference features
        print(f"Running {method} with {combination_name} features...")

        command = base_command_XGBoost_or_MLP(classes_names=classes_names,
                                    splits=config["xgboost"][f"train_{target}"]["splits"] if method == "xgboost" else None,
                                    grid_search=config['xgboost'][f"train_{target}"]['hyperparameters_search_space'] if method == "xgboost" else None,
                                    njobs=config["xgboost"][f"train_{target}"]["njobs"] if method == "xgboost" else None, 
                                    suffix="SECA201", features=circumference, combination_name="circumference",
                                    gait_parameters_path=config['gait_parameters'], **common_parameters)
        
        run_command(command)

        print(f"{method} finished")


def main(args):

    config = load_config(args.config)

    os.chdir("../code/train/")

    for target in args.targets:

        for method in args.methods:

            run_experiments(config, method.lower(), target.capitalize(), args.device)

            

if __name__ == "__main__":

    args = parse_args()

    main(args)


