import subprocess
import argparse
import json
import os

def parse_args():

    parser = argparse.ArgumentParser(description="Technical validation of the HealthGait dataset.")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--device", type=int, required=True, help="Device to run the experiments")
    parser.add_argument("--targets", nargs='+', required=True, help="list with the targets to estimate")
    parser.add_argument('--methods', nargs='+', required=True, help='list with the clases names to classifier')

    return parser.parse_args()


def load_config(config_path):
    
    with open(config_path, "r") as f:

        config = json.load(f)

    return config


def base_command_MoviNet(partitions_path, data_path, patients_measures, 
                         save_dir, device, target, model_id, data_type, partition, 
                         id_experiment, seed, num_frames, img_size, batch_size,
                         learning_rate, epochs, wandb_project, classes_names=None,
                         units=None, optical_flow_method=None, data_class=None):
    
    experiment_name = f"{id_experiment}: Data_type {data_type}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"
    
    base_command = f"python train_MoviNet_{'classification' if target == 'sex' else 'regression'}.py --partitions_file {partitions_path}/partition_{partition}.json " \
                    f"--data_path {data_path} " \
                    f"--patients_info {patients_measures} " \
                    f"--num_frames {num_frames} " \
                    f"--img_size {img_size} " \
                    f"--batch_size {batch_size} " \
                    f"--model_id {model_id} " \
                    f"--learning_rate {learning_rate} " \
                    f"--id_partition {partition} " \
                    f"--save_dir {os.path.join(save_dir, "MoviNet")} " \
                    f"--epochs {epochs} " \
                    f"--device {device} " \
                    f"--data_class {data_class} " \
                    f"--data_type {data_type} " \
                    f"--id_experiment {id_experiment} " \
                    f"--seed {seed} " \
                    f"--wandb_project {wandb_project} " \
                    f"--target {target} " \
                    f"--experiment_name \"{experiment_name}\""
                        
                    
    if target == 'sex':
        base_command += f" --classes_names {classes_names}"
    
    else:
        base_command += f" --units {units}"

    if optical_flow_method:
        base_command += f" --optical_flow_method {optical_flow_method}"


    return base_command


def base_command_XGBoost_or_MLP(partitions_path, data_path, patients_measures):

    pass

def run_command(command):

    subprocess.run(command, shell=True)



def run_experiments(config, device, method, target):

    data_path = config["data_path"]
    save_dir = config["save_dir"]
    partitions_path = os.path.join(config["partitions_path"], target)
    patients_measures = config["patients_measures"]

    # Classification and regression movinet experiments 
    if method == "MoviNet":

        common_parameters = {
            "partitions_path": partitions_path,
            "data_path": data_path,
            "patients_measures": patients_measures,
            "save_dir": save_dir,
            "device": device,
            "target": target,
            "seed": config["MoviNet"][f"train_{target}"]["seed"],
            "epochs": config["MoviNet"][f"train_{target}"]["epochs"],
            "img_size": config["MoviNet"][f"train_{target}"]["img_size"],
            "num_frames": config["MoviNet"][f"train_{target}"]["num_frames"],
            "batch_size": config["MoviNet"][f"train_{target}"]["batch_size"],
            "learning_rate": config["MoviNet"][f"train_{target}"]["learning_rate"],
            "id_experiment": config["MoviNet"][f"train_{target}"]["id_experiment"],
            "wandb_project": config["MoviNet"][f"train_{target}"]["wandb_project"]
        }

        for model_id in ['a0', 'a5']:
            for data_type in ['silhouette', 'semantic_segmentation', 'optical_flow']:
                for data_class in ['WoJ', 'WJ', 'both']:
                    for partition in range(4):

                        print(f"Running MoviNet with target {target}, model_id {model_id}, data_type {data_type}, data_class {data_class}, partition {partition}...")

                        # Adjust command based on target and data type
                        if target == "sex":
                            classes_names = ["woman", "man"]
                        else:
                            units = config["Movinet"][f"train_{target}"]["units"]
                            
                        optical_flow_methods = ['GMFLOW', 'TVL1'] if data_type == 'optical_flow' else [None]
                        
                        for optical_flow_method in optical_flow_methods:
                            command = base_command_MoviNet(
                                classes_names=classes_names if target == 'sex' else None,
                                units=units if target != 'sex' else None,
                                model_id=model_id, data_type=data_type if optical_flow_method is None else f"{data_type} {optical_flow_method}",
                                partition=partition, optical_flow_method=optical_flow_method, data_class=data_class, **common_parameters
                            )
                            run_command(command)

                            print("MoviNet finished")

    # XGBoost and MLP experiments
    elif method == "XGBoost" or method == "MLP":

        fast_gait = "Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
        usual_gait = "Step_UGS Stride_UGS Cadence_UGS Velocity_UGS"
        circumference = "WaistC HipC NeckC"

        combinations = {
            fast_gait: "fast_gait",
            usual_gait: "usual_gait",
            circumference: "circumference",
            f"{fast_gait} {circumference}": "fast_gait_circumference",
            f"{usual_gait} {circumference}": "usual_gait_circumference",
            f"{fast_gait} {usual_gait}": "fast_usual_gait",
            f"{fast_gait} {usual_gait} {circumference}": "fast_usual_gait_circumference"
        }

        for features, combination_name in combinations.items():
            for gait_parameters_path in [config['gait_parameters'], config['gait_parameters_estimation']]:
                
                print(f"Running {method} with {combination_name} features...")

                if method == "XGBoost":

                    run_XGBoost(config, features, gait_parameters_path)

                elif method == "MLP":

                    run_MLP(config, features, gait_parameters_path)

                print(f"{method} finished")



# def run_experiments(config, device, method, target):

#     print(f"Running {target} classification...")

#     data_path = config["data_path"]
#     save_dir = config["save_dir"]
#     partitions_path = os.path.join(config["partitions_path"], target)
#     patients_measures = config["patients_measures"]
#     #gait_parameters = config['gait_parameters']
#     #gait_parameters_estimation = config['gait_parameters_estimation']

#     if method == "MoviNet":

#         seed = config["MoviNet"]["train_sex"]["seed"]
#         epochs = config["MoviNet"]["train_sex"]["epochs"]
#         img_size = config["MoviNet"]["train_sex"]["img_size"]
#         num_frames = config["MoviNet"]["train_sex"]["num_frames"]
#         batch_size = config["MoviNet"]["train_sex"]["batch_size"]
#         learning_rate = config["MoviNet"]["train_sex"]["learning_rate"]
#         id_experiment = config["MoviNet"]["train_sex"]["id_experiment"]
#         wandb_project = config["MoviNet"]["train_sex"]["wandb_project"]

#         movinet_save_dir = os.path.join(save_dir, "MoviNet")

#         for model_id in ['a0', 'a5']:
#             for data_type in ['silhouette', 'semantic_segmentation']:
#                 for data_class in ['WoJ', 'WJ', 'both']:
#                     for partition in range(4):

#                         experiment_name = f"{id_experiment}: Data_type {data_type}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"

#                         print(f"Running MoviNet with model_id {model_id}, data_type {data_type}, data_class {data_class}, partition {partition}...")

#                         if target == "sex":

#                             classes_names = ["woman", "man"]

#                             command = f"python train_MoviNet_classification.py --partitions_file {partitions_path}/partition_{partition}.json " \
#                                         f"--data_path {data_path} " \
#                                         f"--patients_info {patients_measures} " \
#                                         f"--num_frames {num_frames} " \
#                                         f"--img_size {img_size} " \
#                                         f"--batch_size {batch_size} " \
#                                         f"--model_id {model_id} " \
#                                         f"--learning_rate {learning_rate} " \
#                                         f"--id_partition {partition} " \
#                                         f"--save_dir {movinet_save_dir} " \
#                                         f"--epochs {epochs} " \
#                                         f"--device {device} " \
#                                         f"--data_class {data_class} " \
#                                         f"--data_type {data_type} " \
#                                         f"--id_experiment {id_experiment} " \
#                                         f"--seed {seed} " \
#                                         f"--wandb_project {wandb_project} " \
#                                         f"--classes_names {classes_names} " \
#                                         f"--target {target} " \
#                                         f"--experiment_name \"{experiment_name}\" "
                            
#                         else:

#                             units = config["Movinet"][f"train_{target}"]["units"]
                         
#                             command = f"python train_MoviNet_regression.py --partitions_file {partitions_path}/partition_{partition}.json " \
#                                         f"--data_path {data_path} " \
#                                         f"--patients_info {patients_measures} " \
#                                         f"--num_frames {num_frames} " \
#                                         f"--img_size {img_size} " \
#                                         f"--batch_size {batch_size} " \
#                                         f"--units {units} " \
#                                         f"--model_id {model_id} " \
#                                         f"--learning_rate {learning_rate} " \
#                                         f"--id_partition {partition} " \
#                                         f"--save_dir {movinet_save_dir} " \
#                                         f"--epochs {epochs} " \
#                                         f"--device {device} " \
#                                         f"--data_class {data_class} " \
#                                         f"--data_type {data_type} " \
#                                         f"--id_experiment {id_experiment} " \
#                                         f"--seed {seed} " \
#                                         f"--wandb_project {wandb_project} " \
#                                         f"--target {target} " \
#                                         f"--experiment_name {experiment_name} "


#                         run_command(command)


#                         print("MoviNet finished")
    
                    
#         for optical_flow_method in ['GMFLOW', 'TVL1']:
#             for data_class in ['WoJ', 'WJ', 'both']:
#                 for partition in range(4):

#                     experiment_name = f"{id_experiment}: Data_type {data_type} {optical_flow_method}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"

#                     print(f"Running MoviNet with optical_flow_method {optical_flow_method}, data_class {data_class}, partition {partition}...")

#                     if target == "sex":

#                         classes_names = ["woman", "man"]

#                         command = f"python train_MoviNet_classification.py --partitions_file {partitions_path}/partition_{partition}.json " \
#                                     f"--data_path {data_path} " \
#                                     f"--patients_info {patients_measures} " \
#                                     f"--num_frames {num_frames} " \
#                                     f"--img_size {img_size} " \
#                                     f"--batch_size {batch_size} " \
#                                     f"--model_id {model_id} " \
#                                     f"--learning_rate {learning_rate} " \
#                                     f"--id_partition {partition} " \
#                                     f"--save_dir {movinet_save_dir} " \
#                                     f"--epochs {epochs} " \
#                                     f"--device {device} " \
#                                     f"--data_class {data_class} " \
#                                     f"--data_type {data_type} " \
#                                     f"--id_experiment {id_experiment} " \
#                                     f"--seed {seed} " \
#                                     f"--wandb_project {wandb_project} " \
#                                     f"--classes_names {classes_names} " \
#                                     f"--target {target} " \
#                                     f"--experiment_name \"{experiment_name}\" " \
#                                     f"--optical_flow_method {optical_flow_method}"
                        
#                     else:

#                         units = config["Movinet"][f"train_{target}"]["units"]
                        
#                         command = f"python train_MoviNet_regression.py --partitions_file {partitions_path}/partition_{partition}.json " \
#                                     f"--data_path {data_path} " \
#                                     f"--patients_info {patients_measures} " \
#                                     f"--num_frames {num_frames} " \
#                                     f"--img_size {img_size} " \
#                                     f"--batch_size {batch_size} " \
#                                     f"--units {units} " \
#                                     f"--model_id {model_id} " \
#                                     f"--learning_rate {learning_rate} " \
#                                     f"--id_partition {partition} " \
#                                     f"--save_dir {movinet_save_dir} " \
#                                     f"--epochs {epochs} " \
#                                     f"--device {device} " \
#                                     f"--data_class {data_class} " \
#                                     f"--data_type {data_type} " \
#                                     f"--id_experiment {id_experiment} " \
#                                     f"--seed {seed} " \
#                                     f"--wandb_project {wandb_project} " \
#                                     f"--target {target} " \
#                                     f"--experiment_name {experiment_name} " \
#                                     f"--optical_flow_method {optical_flow_method}"


#                     print("MoviNet finished")



# def run_classification(method, config, device):

#     print("Running sex classification...")

#     data_path = config["data_path"]
#     save_dir = config["save_dir"]
#     partitions_path = os.path.join(config["partitions_path"], "sex")
#     patients_measures = config["patients_measures"]
#     gait_parameters = config['gait_parameters']
#     gait_parameters_estimation = config['gait_parameters_estimation']

#     target = "Sex"
#     classes_names = ["woman", "man"]
#     n_decimals = 3

#     def run_MoviNet(config, data_type, optical_flow_method, partition, device, data_class, model_id):

#         seed = config["MoviNet"]["train_sex"]["seed"]
#         epochs = config["MoviNet"]["train_sex"]["epochs"]
#         img_size = config["MoviNet"]["train_sex"]["img_size"]
#         num_frames = config["MoviNet"]["train_sex"]["num_frames"]
#         batch_size = config["MoviNet"]["train_sex"]["batch_size"]
#         learning_rate = config["MoviNet"]["train_sex"]["learning_rate"]
#         id_experiment = config["MoviNet"]["train_sex"]["id_experiment"]
#         wandb_project = config["MoviNet"]["train_sex"]["wandb_project"]

#         experiment_name = f"{id_experiment}: Data_type {data_type} {optical_flow_method}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"
        
#         opt_flow_param = f"--optical_flow_method {optical_flow_method}" if data_type == "optical_flow" else ""

#         movinet_save_dir = os.path.join(save_dir, "MoviNet")

#         subprocess.run(f"python train_MoviNet_classification.py --partitions_file {partitions_path}/partition_{partition}.json "
#                     f"--data_path {data_path} "
#                     f"--patients_info {patients_measures} "
#                     f"--num_frames {num_frames} "
#                     f"--img_size {img_size} "
#                     f"--batch_size {batch_size} "
#                     f"--model_id {model_id} "
#                     f"--learning_rate {learning_rate} "
#                     f"--id_partition {partition} "
#                     f"--save_dir {movinet_save_dir} "
#                     f"--epochs {epochs} "
#                     f"--device {device} "
#                     f"--data_class {data_class} "
#                     f"--data_type {data_type} "
#                     f"--id_experiment {id_experiment} "
#                     f"--seed {seed} "
#                     f"--wandb_project {wandb_project} "
#                     f"--classes_names {classes_names} "
#                     f"--target {target} "
#                     f"--experiment_name \"{experiment_name}\" "
#                     f"{opt_flow_param}", shell=True)


#     def run_XGBoost(config, features, gait_parameters_path):

#         seed = config["XGBoost"]["train_sex"]["seed"]
#         njobs = config["XGBoost"]["train_sex"]["njobs"]
#         splits = config["XGBoost"]["train_sex"]["splits"]
#         grid_search = config['XGBoost']['train_sex']['hyperparameters_search_space']

#         combination_name = combinations[features]

#         if gait_parameters in gait_parameters_path:
#             suffix = "OptoGait/"
#         else:
#             suffix = "Estimated/"

#         hyperparameters_output = f"{save_dir}/XGBoost/sex_classification/hyperparameters/{combination_name}/{suffix}"
#         evaluation_output = f"{save_dir}/XGBoost/sex_classification/evaluation/{combination_name}/{suffix}"


#         subprocess.run(f"python train_xgboost_classification.py "
#                        f"--gait_parameters {gait_parameters_path} "
#                        f"--patients_measures {patients_measures} "
#                        f"--partitions_path {partitions_path} "
#                        f"--features {features} "
#                        f"--target {target} "
#                        f"--seed {str(seed)} "
#                        f"--splits {str(splits)} "
#                        f"--hyperparameters_path {hyperparameters_output} "
#                        f"--classes_names \"{classes_names}\" "
#                        f"--evaluation_path {evaluation_output} "
#                        f"--grid_search {grid_search} "
#                        f"--njobs {str(njobs)} "
#                        f"--n_decimals {str(n_decimals)}", shell=True)


#     def run_MLP(config, features, gait_parameters_path):

#         seed = config["XGBoost"]["train_sex"]["seed"]

#         combination_name = combinations[features]

#         if gait_parameters in gait_parameters_path:
#             suffix = "OptoGait/"
#         else:
#             suffix = "Estimated/"

#         hyperparameters_output = f"{save_dir}/MLP/hyperparameters/{combination_name}/{suffix}"
#         evaluation_output = f"{save_dir}/MLP/evaluation/{combination_name}/{suffix}"

#         subprocess.run(f"python train_mlp_classification.py "
#                    f"--gait_parameters {gait_parameters_path} "
#                    f"--patients_measures {patients_measures} "
#                    f"--partitions_path {partitions_path} "
#                    f"--features {features} "
#                    f"--target {target} "
#                    f"--seed {seed} "
#                    f"--hyperparameters_path {hyperparameters_output} "
#                    f"--evaluation_path {evaluation_output} "
#                    f"--n_decimals {n_decimals} "
#                    f"--classes_names {classes_names}", shell=True)
        
    
#     if method == "MoviNet":

#         for model_id in ['a0', 'a5']:
#             for data_type in ['silhouette', 'semantic_segmentation']:
#                 for data_class in ['WoJ', 'WJ', 'both']:
#                     for partition in range(4):

#                         print(f"Running MoviNet with model_id {model_id}, data_type {data_type}, data_class {data_class}, partition {partition}...")

#                         run_MoviNet(config, data_type, "", partition, device, data_class, model_id)

#                         print("MoviNet finished")

                        

#         for optical_flow_method in ['GMFLOW', 'TVL1']:
#             for data_class in ['WoJ', 'WJ', 'both']:
#                 for partition in range(4):

#                     if method == "MoviNet":

#                         print(f"Running MoviNet with optical_flow_method {optical_flow_method}, data_class {data_class}, partition {partition}...")

#                         run_MoviNet(config, 'optical_flow', optical_flow_method, partition, device, data_class, model_id)

#                         print("MoviNet finished")

#     elif method == "XGBoost" or method == "MLP":

#         fast_gait = "Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
#         usual_gait = "Step_UGS Stride_UGS Cadence_UGS Velocity_UGS"
#         circumference = "WaistC HipC NeckC"

#         combinations = {
#             fast_gait: "fast_gait",
#             usual_gait: "usual_gait",
#             circumference: "circumference",
#             f"{fast_gait} {circumference}": "fast_gait_circumference",
#             f"{usual_gait} {circumference}": "usual_gait_circumference",
#             f"{fast_gait} {usual_gait}": "fast_usual_gait",
#             f"{fast_gait} {usual_gait} {circumference}": "fast_usual_gait_circumference"
#         }

#         for features, combination_name in combinations.items():
#             for gait_parameters_path in [gait_parameters, gait_parameters_estimation]:
                
#                 print(f"Running {method} with {combination_name} features...")

#                 if method == "XGBoost":

#                     run_XGBoost(config, features, gait_parameters_path)

#                 elif method == "MLP":

#                     run_MLP(config, features, gait_parameters_path)

#                 print(f"{method} finished")
            
    

# def run_regression(method, target, config, device):
    
#     print(f"Running {target} regression...")

#     data_path = config["data_path"]
#     save_dir = config["save_dir"]
#     partitions_path = os.path.join(config["partitions_path"], f"{target}")
#     patients_measures = config["patients_measures"]
#     gait_parameters = config['gait_parameters']
#     gait_parameters_estimation = config['gait_parameters_estimation']

#     n_decimals = 3

#     def run_MoviNet(config, data_type, optical_flow_method, partition, device, data_class, model_id):

#         seed = config["Movinet"][f"train_{target}"]["seed"]
#         epochs = config["Movinet"][f"train_{target}"]["epochs"]
#         img_size = config["Movinet"][f"train_{target}"]["img_size"]
#         num_frames = config["Movinet"][f"train_{target}"]["num_frames"]
#         batch_size = config["Movinet"][f"train_{target}"]["batch_size"]
#         learning_rate = config["Movinet"][f"train_{target}"]["learning_rate"]
#         units = config["Movinet"][f"train_{target}"]["units"]
#         id_experiment = config["Movinet"][f"train_{target}"]["id_experiment"]
#         wandb_project = config["Movinet"][f"train_{target}"]["wandb_project"]

#         experiment_name = f"{id_experiment}: Data_type {data_type} {optical_flow_method}, Data_Class {data_class}, Num_Frames {num_frames}, Model_ID {model_id}, SEED {seed}"
        
#         opt_flow_param = f"--optical_flow_method {optical_flow_method}" if data_type == "optical_flow" else ""

#         movinet_save_dir = os.path.join(save_dir, "MoviNet")

#         subprocess.run(f"python train_MoviNet_regression.py --partitions_file {partitions_path}/partition_{partition}.json "
#                        f"--data_path {data_path} "
#                        f"--patients_info {patients_measures} "
#                        f"--num_frames {num_frames} "
#                        f"--img_size {img_size} "
#                        f"--batch_size {batch_size} "
#                        f"--units {units} "
#                        f"--model_id {model_id} "
#                        f"--learning_rate {learning_rate} "
#                        f"--id_partition {partition} "
#                        f"--save_dir {movinet_save_dir} "
#                        f"--epochs {epochs} "
#                        f"--device {device} "
#                        f"--data_class {data_class} "
#                        f"--data_type {data_type} "
#                        f"--id_experiment {id_experiment} "
#                        f"--seed {seed} "
#                        f"--wandb_project {wandb_project} "
#                        f"--target {target} "
#                        f"--experiment_name {experiment_name} "
#                        f"{opt_flow_param}", shell=True)


#     def run_XGBoost(config, features, gait_parameters_path):

#         seed = config["XGBoost"][f"train_{target}"]["seed"]
#         njobs = config["XGBoost"][f"train_{target}"]["njobs"]
#         splits = config["XGBoost"][f"train_{target}"]["splits"]
#         grid_search = config['XGBoost'][f"train_{target}"]['hyperparameters_search_space']

#         combination_name = combinations[features]

#         if gait_parameters in gait_parameters_path:
#             suffix = "OptoGait/"
#         else:
#             suffix = "Estimated/"

#         hyperparameters_output = f"{save_dir}/XGBoost/{target}_regression/hyperparameters/{combination_name}/{suffix}"
#         evaluation_output = f"{save_dir}/XGBoost/{target}_regression/evaluation/{combination_name}/{suffix}"

#         subprocess.run(f"python train_xgboost_regression.py --gait_parameters {gait_parameters_path} "
#                         f"--patients_measures {patients_measures} "
#                         f"--partitions_path {partitions_path} "
#                         f"--features {features} "
#                         f"--target {target} "
#                         f"--seed {str(seed)} "
#                         f"--splits {str(splits)} "
#                         f"--hyperparameters_path {hyperparameters_output} "
#                         f"--evaluation_path {evaluation_output} "
#                         f"--grid_search {grid_search} "
#                         f"--njobs {str(njobs)}",
#                         f"--n_decimals {n_decimals}", shell=True)
        
#     def run_MLP(config, features, gait_parameters_path):

#         seed = config["MLP"][f"train_{target}"]["seed"]

#         combination_name = combinations[features]

#         if gait_parameters in gait_parameters_path:
#             suffix = "OptoGait/"
#         else:
#             suffix = "Estimated/"

#         hyperparameters_output = f"{save_dir}/MLP/{target}_regression/hyperparameters/{combination_name}/{suffix}"
#         evaluation_output = f"{save_dir}/MLP/{target}_regression/evaluation/{combination_name}/{suffix}"

#         subprocess.run(f"python train_mlp_regression.py --gait_parameters {gait_parameters_path} "
#                        f"--patients_measures {patients_measures} "
#                        f"--partitions_path {partitions_path} "
#                        f"--features {features} "
#                        f"--target {target} "
#                        f"--seed {seed} ",
#                        f"--hyperparameters_path {hyperparameters_output} "
#                        f"--evaluation_path {evaluation_output} "
#                        f"--n_decimals {n_decimals}", shell=True)


#     if method == "MoviNet":

#         for model_id in ['a0', 'a5']:
#             for data_type in ['silhouette', 'semantic_segmentation']:
#                 for data_class in ['WoJ', 'WJ', 'both']:
#                     for partition in range(4):

#                         print(f"Running MoviNet with model_id {model_id}, data_type {data_type}, data_class {data_class}, partition {partition}...")

#                         run_MoviNet(config, data_type, "", partition, device, data_class, model_id)

#                         print("MoviNet finished")

                        
#         for optical_flow_method in ['GMFLOW', 'TVL1']:
#             for data_class in ['WoJ', 'WJ', 'both']:
#                 for partition in range(4):

#                     if method == "MoviNet":

#                         print(f"Running MoviNet with optical_flow_method {optical_flow_method}, data_class {data_class}, partition {partition}...")

#                         run_MoviNet(config, 'optical_flow', optical_flow_method, partition, device, data_class, model_id)

#                         print("MoviNet finished")

#     elif method == "XGBoost" or method == "MLP":

#         fast_gait = "Step_FGS Stride_FGS Cadence_FGS Velocity_FGS"
#         usual_gait = "Step_UGS Stride_UGS Cadence_UGS Velocity_UGS"
#         circumference = "WaistC HipC NeckC"

#         combinations = {
#             fast_gait: "fast_gait",
#             usual_gait: "usual_gait",
#             circumference: "circumference",
#             f"{fast_gait} {circumference}": "fast_gait_circumference",
#             f"{usual_gait} {circumference}": "usual_gait_circumference",
#             f"{fast_gait} {usual_gait}": "fast_usual_gait",
#             f"{fast_gait} {usual_gait} {circumference}": "fast_usual_gait_circumference"
#         }

#         for features, combination_name in combinations.items():
#             for gait_parameters_path in [gait_parameters, gait_parameters_estimation]:
                
#                 print(f"Running {method} with {combination_name} features...")

#                 if method == "XGBoost":

#                     run_XGBoost(config, features, gait_parameters_path)

#                 elif method == "MLP":

#                     run_MLP(config, features, gait_parameters_path)

#                 print(f"{method} finished")



def main(args):

    config = load_config(args.config)

    os.chdir("../train/")

    for target in args.targets:

        for method in args.methods:

            run_experiments(config, args.device, method, target, args.device)
            
            # # TODO: Hacer las comprobaciones con la cadena target en minúsculas
            # if target == "sex":

            #     run_classification(method, config, DEVICE)

            # else:

            #     run_regression(method, target, config, DEVICE)
        

if __name__ == "__main__":

    args = parse_args()

    main(args)


