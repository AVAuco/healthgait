import os
import json
import random
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error


import keras_tuner


import warnings

warnings.filterwarnings("ignore")

def parse_opt():

    parser = argparse.ArgumentParser(
        description = 'Train xgboost'
    )

    parser.add_argument('--gait_parameters', type = str, required = True, help = 'gait parameters path')
    parser.add_argument('--patients_measures', type = str, required = True, help = 'patients measures path')
    parser.add_argument('--partitions_path', type = str, required = True, help = 'partitions path')
    parser.add_argument('--features', nargs='+', required=True, help = 'features used to train')
    parser.add_argument('--target', type = str, required = True, help = 'class')
    parser.add_argument('--hyperparameters_path', type = str, required = True, help = 'hyperparameters path')
    parser.add_argument('--evaluation_path', type = str, required = True, help = 'path where to save model evaluation (metrics and cm)')
    parser.add_argument('--n_decimals', type = int, required = True, help = 'number of decimals to round results')
    parser.add_argument('--seed', type = int, default = 27, help = "seed")

    args = parser.parse_args()

    return args



def set_seed(seed: int = 42) -> None:
  
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  tf.keras.utils.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")

def get_data(GAIT_PATH, MEASURES_PATH, PARTITIONS_FILE, features, target):
    

    df1 = pd.read_csv(GAIT_PATH)
    df2 = pd.read_csv(MEASURES_PATH)

    df = pd.merge(df1, df2, on='ID')

    #print(df.head())

    # Lectura desde fichero de los pacientes utilizados para el entrenamiento.
    with open(PARTITIONS_FILE, 'r') as f:
    # Lee el contenido del archivo
        partitions_data = json.load(f)

    train_patients = partitions_data["train"]
    val_patients = partitions_data["validation"]
    test_patients = partitions_data["test"]

    train_data = df[df['ID'].isin(train_patients)]
    X_train = train_data[features].to_numpy()
    y_train = train_data[target].to_numpy()

    val_data = df[df['ID'].isin(val_patients)]
    X_val = val_data[features].to_numpy()
    y_val = val_data[target].to_numpy()

    test_data = df[df['ID'].isin(test_patients)]
    X_test = test_data[features].to_numpy()
    y_test = test_data[target].to_numpy()

    data = {
        "train": {"data": X_train, "target": y_train},
        "val": {"data": X_val, "target": y_val},
        "test": {"data": X_test, "target": y_test}
    }

    return data


# Tratamiento de los valores perdidos y normalización de los datos
def preprocess_data(data):

    # MISSING VALUES
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(data["train"]["data"])

    data["train"]["data"] = imp.transform(data["train"]["data"])
    data["val"]["data"] = imp.transform(data["val"]["data"])
    data["test"]["data"] = imp.transform(data["test"]["data"])

    # NORMALIZATION
    
    
    scaler = preprocessing.StandardScaler()

    scaler.fit(data["train"]["data"])

    data["train"]["data"] = scaler.transform(data["train"]["data"])
    data["val"]["data"] = scaler.transform(data["val"]["data"])
    data["test"]["data"] = scaler.transform(data["test"]["data"])

    target_scaler = preprocessing.MinMaxScaler()

    for key in ['train', 'val', 'test']:
        # Remodelar los datos
        targets_reshaped = data[key]["target"].reshape(-1, 1)

        # Ajustar el escalador solo en el conjunto de entrenamiento
        if key == 'train':
            target_scaler.fit(targets_reshaped)

        # Transformar los datos
        data[key]["target"] = target_scaler.transform(targets_reshaped)


    # normalizer = preprocessing.Normalizer()
    # normalizer.fit(data["train"]["data"])

    # data["train"]["data"] = normalizer.transform(data["train"]["data"])
    # data["test"]["data"] = normalizer.transform(data["test"]["data"])

    return data, target_scaler


def build_model(hp):

    model = tf.keras.Sequential()

    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            tf.keras.layers.Dense(
                # Tune number of units separately.
                units = hp.Int(f"units_{i}", min_value = 32, max_value = 512, step=32),
                activation = hp.Choice("activation", ["relu", "tanh"]),
                kernel_initializer = "normal")
            )

    if hp.Boolean("dropout"):

        model.add(tf.keras.layers.Dropout(rate = 0.25))

    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    lr = hp.Float("lr", min_value = 1e-4, max_value = 1e-2, sampling = "log")

    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = lr), metrics = ["mean_absolute_error"])
    #model.compile(loss = 'mean_absolute_error', optimizer = tf.keras.optimizers.Adam(learning_rate = lr), metrics = ["mean_squared_error"])

    return model

def evaluate_model_by_age_groups(y_true, y_pred, age_groups):
    group_errors = {group: [] for group in age_groups}

    for true_age, pred_age in zip(y_true, y_pred):
        for group, (age_min, age_max) in age_groups.items():
            if age_min <= true_age < age_max:
                group_errors[group].append(abs(true_age - pred_age))
                break

    group_mae = {group: np.mean(errors) for group, errors in group_errors.items()}
    return group_mae


def main(args):

    GAIT_PATH = args.gait_parameters
    MEASURES_PATH = args.patients_measures
    PARTITIONS_PATH = args.partitions_path
    HYPERPARAMETERS_PATH = args.hyperparameters_path
    EVALUATION_PATH = args.evaluation_path
    FEATURES = args.features 
    TARGET = args.target
    SEED = args.seed
    N_DECIMALS = args.n_decimals

    results = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "manual_mae": [],
        "std_mae": [],
        "partition_order": []
    }

    age_groups = {
        "18-34": (18, 35),
        "35-49": (35, 50),
        "50-64": (50, 65)
    }

    results_by_age = {
        "18-34": [],
        "35-49": [],
        "50-64": []
    }

    set_seed(SEED)

    for partition_file in os.listdir(PARTITIONS_PATH):
    #for partition_file in sorted(os.listdir(PARTITIONS_PATH)):

        #set_seed(SEED)

        data = get_data(GAIT_PATH, MEASURES_PATH, os.path.join(PARTITIONS_PATH, partition_file), FEATURES, TARGET)

        data, target_scaler = preprocess_data(data)

        tuner = keras_tuner.BayesianOptimization(hypermodel = build_model,
                                         objective = 'val_loss',
                                         max_trials = 10,
                                         overwrite = True,
                                         directory = pathlib.Path(HYPERPARAMETERS_PATH).parent,
                                         project_name = pathlib.Path(HYPERPARAMETERS_PATH).stem,
                                         seed = SEED)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 100)

        callbacks = [early_stopping]

        tuner.search(data["train"]["data"], data["train"]["target"], epochs = 1000, validation_data = (data["val"]["data"], data["val"]["target"]), callbacks = callbacks)

        models = tuner.get_best_models(num_models=2)
        best_model = models[0]

        # Reentrenar el modelo utilizando tambien el conjunto de validación

        # best_hps = tuner.get_best_hyperparameters(5)

        # best_model = build_model(best_hps[0])

        # x_all = np.concatenate((data["train"]["data"], data["val"]["data"]))
        # y_all = np.concatenate((data["train"]["target"], data["val"]["target"]))

        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "train_loss", mode = "min", patience = 100)

        # callbacks = [early_stopping]

        # best_model.fit(x=x_all, y=y_all, epochs=1000, callbacks = callbacks)

        # Model Evaluation

        y_preds = best_model.predict(data["test"]["data"])

        y_preds = target_scaler.inverse_transform(y_preds.reshape(-1, 1))

        data["test"]["target"] = target_scaler.inverse_transform(data["test"]["target"].reshape(-1, 1))

        age_group_errors = evaluate_model_by_age_groups(data["test"]["target"], y_preds, age_groups)

        results_by_age["18-34"].append(age_group_errors["18-34"])
        results_by_age["35-49"].append(age_group_errors["35-49"]) 
        results_by_age["50-64"].append(age_group_errors["50-64"])

        pathlib.Path(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem)).mkdir(parents = True, exist_ok = True)

        # Metrics

        total = []

        for y_pred, y_true in zip(y_preds, data["test"]["target"]):

            total.append(abs(y_pred - y_true))
        
        mse = round(mean_squared_error(data["test"]["target"], y_preds), N_DECIMALS)

        results["mse"].append(mse)
        results["rmse"].append(np.sqrt(mse))
        results["mae"].append(round(mean_absolute_error(data["test"]["target"], y_preds), N_DECIMALS))
        results["manual_mae"].append(round(np.mean(total), N_DECIMALS))
        results["std_mae"].append(round(np.std(total), N_DECIMALS))
        results["partition_order"].append(partition_file)


        # Relative error

        data = {"real_value": data["test"]["target"].flatten(), "predicted_value": y_preds.flatten()}

        df = pd.DataFrame(data)

        df["absolute_error"] = abs(df["real_value"] - df["predicted_value"])

        df["relative_error"] = (df["absolute_error"] / df["real_value"]) * 100


        df.sort_values(by=['relative_error'], ascending = False).to_csv(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "errors.csv"), index = False)

        plt.hist(x=df["relative_error"], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
        
        plt.yticks(fontsize = 15)
        plt.xticks(fontsize = 15)
        
        plt.xlabel('Relative error percentage', fontsize = 15)
        plt.ylabel('Frequency', fontsize = 15)
        plt.title(f'Relative error distribution for {TARGET.lower()} estimation', fontsize = 15)
        
        plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "relative_error_distribution_weight.svg"))

        plt.cla()

    # Get mean results and save in EVALUATION_PATH
        
    results["mean_mse"] = round(np.mean(results["mse"]), N_DECIMALS)
    results["mean_mae"] = round(np.mean(results["mae"]), N_DECIMALS)
    results["mean_std_mae"] = round(np.mean(results["std_mae"]), N_DECIMALS)
        
    with open(os.path.join(EVALUATION_PATH, f'results.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results, f, ensure_ascii = False, indent = 4)


    results_by_age["18-34"] = np.mean(results_by_age["18-34"])
    results_by_age["35-49"] = np.mean(results_by_age["35-49"])
    results_by_age["50-64"] = np.mean(results_by_age["50-64"])

    with open(os.path.join(EVALUATION_PATH, f'age_group_errors.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results_by_age, f, ensure_ascii = False, indent = 4)


    
if __name__ == '__main__':

    args = parse_opt()

    main(args)