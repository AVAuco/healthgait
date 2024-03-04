import os
import json
import random
import argparse
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error


import keras_tuner

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from scipy.special import softmax

from sklearn.metrics import confusion_matrix

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
    parser.add_argument('--classes_names', nargs='+', required=True, help = 'classes name')
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

    def prepare_data(patients):
        data = df[df['ID'].isin(patients)]
        # Elimina filas donde la columna 'target' tiene valores nulos
        data = data.dropna(subset=[target])
        X = data[features].to_numpy()
        y = data[target].to_numpy(dtype=np.uint8)
        return {"data": X, "target": y}

    train_data = prepare_data(train_patients)
    val_data = prepare_data(val_patients)
    test_data = prepare_data(test_patients)

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data
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

    return data


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

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

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
    CLASSES_NAMES = args.classes_names

    set_seed(SEED)

    results = {
        "accuracy": [],
        "f1-score": []
    }

    for partition_file in os.listdir(PARTITIONS_PATH):

        data = get_data(GAIT_PATH, MEASURES_PATH, os.path.join(PARTITIONS_PATH, partition_file), FEATURES, TARGET)

        data = preprocess_data(data)

        tuner = keras_tuner.BayesianOptimization(hypermodel = build_model,
                                         objective = 'val_accuracy',
                                         max_trials = 10,
                                         overwrite = True,
                                         directory = pathlib.Path(HYPERPARAMETERS_PATH).parent,
                                         project_name = pathlib.Path(HYPERPARAMETERS_PATH).stem,
                                         seed = SEED)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", mode = "max", patience = 100)

        callbacks = [early_stopping]

        tuner.search(data["train"]["data"], data["train"]["target"], epochs = 1000, validation_data = (data["val"]["data"], data["val"]["target"]), callbacks = callbacks)

        models = tuner.get_best_models(num_models=2)
        best_model = models[0]

        # Model Evaluation

        y_preds = (best_model.predict(data["test"]["data"]) > 0.5).astype("int32").flatten()

        pathlib.Path(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem)).mkdir(parents = True, exist_ok = True)

        metrics = {
                   "accuray": round(accuracy_score(data["test"]["target"], y_preds), N_DECIMALS), 
                   "f1 macro": round(f1_score(data["test"]["target"], y_preds), N_DECIMALS)
                  }


        results_partition = pd.DataFrame(data = [list(metrics.values())],
        columns = list(metrics.keys()))

        # Almacenar resultados fichero csv

        results_partition.to_csv(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "results.csv"), index = False)

        # Metrics

        results["accuracy"].append(metrics["accuray"])
        results["f1-score"].append(metrics["f1 macro"])

        cm = confusion_matrix(data["test"]["target"], y_preds)

        # Normalise
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))

        # Crear el heatmap
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels = CLASSES_NAMES, yticklabels = CLASSES_NAMES, 
                    cmap='Blues', annot_kws={"size": 30})

        # Aumentar el tamaño de las etiquetas de las clases
        ax.set_xticklabels(CLASSES_NAMES, fontsize=20)  # Ajusta el tamaño de la fuente según sea necesario
        ax.set_yticklabels(CLASSES_NAMES, fontsize=20)  # Ajusta el tamaño de la fuente según sea necesario

        plt.ylabel('Actual', fontsize=14)  # Ajustar el tamaño de la fuente del eje y
        plt.xlabel('Predicted', fontsize=14)  # Ajustar el tamaño de la fuente del eje x

        plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "cm.svg"))

        plt.cla()


    # Get mean results and save in EVALUATION_PATH
        
    results["mean_accuracy"] = round(np.mean(results["accuracy"]), N_DECIMALS)
    results["mean_f1"] = round(np.mean(results["f1-score"]), N_DECIMALS)
        
    with open(os.path.join(EVALUATION_PATH, f'results.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results, f, ensure_ascii = False, indent = 4)


    
if __name__ == '__main__':

    args = parse_opt()

    main(args)