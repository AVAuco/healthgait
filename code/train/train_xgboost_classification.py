import os
import json
import pickle
import argparse
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skopt import BayesSearchCV

import xgboost as xgb


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

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
    parser.add_argument('--classes_names', nargs='+', required=True, help = 'classes name')
    parser.add_argument('--target', type = str, required = True, help = 'class')
    parser.add_argument('--hyperparameters_path', type = str, required = True, help = 'hyperparameters path')
    parser.add_argument('--evaluation_path', type = str, required = True, help = 'path where to save model evaluation (metrics and cm)')
    parser.add_argument('--grid_search', type = str, required = True, help = 'grid search json')
    parser.add_argument('--splits', type = int, required = True, help = 'number of split using in cross-validation')
    parser.add_argument('--njobs', type = int, required = True, help = 'number of jobs to run in parallel')
    parser.add_argument('--n_decimals', type = int, required = True, help = 'decimals used to round the results')
    parser.add_argument('--seed', type = int, default = 27, help = "seed")

    args = parser.parse_args()

    return args



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
    y_train = train_data[target].to_numpy(dtype=np.uint8)

    val_data = df[df['ID'].isin(val_patients)]
    X_val = val_data[features].to_numpy()
    y_val = val_data[target].to_numpy(dtype=np.uint8)

    test_data = df[df['ID'].isin(test_patients)]
    X_test = test_data[features].to_numpy()
    y_test = test_data[target].to_numpy(dtype=np.uint8)

    data = {
        "train": {"data": X_train, "target": y_train},
        "val": {"data": X_val, "target": y_val},
        "test": {"data": X_test, "target": y_test}
    }

    data["test"]["ID"] = test_data['ID'].values

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

    # normalizer = preprocessing.Normalizer()
    # normalizer.fit(data["train"]["data"])

    # data["train"]["data"] = normalizer.transform(data["train"]["data"])
    # data["test"]["data"] = normalizer.transform(data["test"]["data"])

    return data


def main(args):

    GAIT_PATH = args.gait_parameters
    MEASURES_PATH = args.patients_measures
    PARTITIONS_PATH = args.partitions_path
    HYPERPARAMETERS_PATH = args.hyperparameters_path
    EVALUATION_PATH = args.evaluation_path
    CLASSES_NAMES = args.classes_names
    GRID_SEARCH = args.grid_search
    FEATURES = args.features 
    TARGET = args.target
    SPLITS = args.splits
    SEED = args.seed
    NJOBS = args.njobs
    N_DECIMALS = args.n_decimals

    with open(GRID_SEARCH, 'r') as f:
    # Lee el contenido del archivo
        param_grid = json.load(f)


    results = {
        "accuracy": [],
        "f1-score": []
    }

    for partition_file in os.listdir(PARTITIONS_PATH):

        data = get_data(GAIT_PATH, MEASURES_PATH, os.path.join(PARTITIONS_PATH, partition_file), FEATURES, TARGET)

        data = preprocess_data(data)

        model = xgb.XGBClassifier(objective = "binary:logistic", random_state = SEED, early_stopping_rounds = 40)

        cv = StratifiedKFold(n_splits = SPLITS, random_state = SEED, shuffle = True)

        search = BayesSearchCV(model, param_grid, cv = cv, scoring='accuracy', n_jobs = NJOBS, random_state = SEED)

        eval_set = [(data["val"]["data"], data["val"]["target"])]

        np.int = int
        #grid_search.fit(data["train"]["data"], data["train"]["target"], early_stopping_rounds = 10)
        search.fit(data["train"]["data"], data["train"]["target"], eval_set = eval_set, verbose = False)

        pathlib.Path(os.path.join(HYPERPARAMETERS_PATH, pathlib.Path(partition_file).stem)).mkdir(parents = True, exist_ok = True)

        search.best_params_["accuracy"] = search.best_score_

        # Save hyperparameters
        with open(os.path.join(HYPERPARAMETERS_PATH, pathlib.Path(partition_file).stem, f'best_params.json'), 'w', encoding = 'utf-8') as f:

            json.dump(search.best_params_, f, ensure_ascii = False, indent = 4)

        #print(f"Grid Search completado para la partición {partition_file}")

        # Model Evaluation

        y_pred = search.best_estimator_.predict(data["test"]["data"])
        

        # Identificar predicciones incorrectas y sus IDs
        incorrect_predictions_mask = data["test"]["target"] != y_pred
        incorrect_ids = data["test"]['ID'][incorrect_predictions_mask]
        incorrect_true_classes = data["test"]["target"][incorrect_predictions_mask]
        incorrect_pred_classes = y_pred[incorrect_predictions_mask]

        incorrect_predictions = {
            "ID": incorrect_ids.tolist(),
            "True Class": incorrect_true_classes.tolist(),
            "Predicted Class": incorrect_pred_classes.tolist()
        }

        pathlib.Path(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem)).mkdir(parents = True, exist_ok = True)

        json_s = json.dumps(incorrect_predictions)

        with open(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "incorrect_predictions.json"), 'w') as f:

            f.write(json_s)

        # Metrics

        accuracy = accuracy_score(data["test"]["target"], y_pred)
        f1 = f1_score(data["test"]["target"], y_pred)

        results["accuracy"].append(accuracy)
        results["f1-score"].append(f1)

        # Confusión matrix

        cm = confusion_matrix(data["test"]["target"], y_pred)

        # Normalise
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))

        sns.set(font_scale=3.0) # Adjust to fit

        # Crear el heatmap
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=CLASSES_NAMES, yticklabels=CLASSES_NAMES, 
                    cmap='Blues', annot_kws={"size": 30})

        # Aumentar el tamaño de las etiquetas de las clases
        ax.set_xticklabels(CLASSES_NAMES, fontsize=30)  # Ajusta el tamaño de la fuente según sea necesario
        ax.set_yticklabels(CLASSES_NAMES, fontsize=30)  # Ajusta el tamaño de la fuente según sea necesario

        plt.ylabel('Actual', fontsize=30, labelpad=15)
        plt.xlabel('Predicted', fontsize=30, labelpad=15)

        plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "cm.svg"))

        plt.cla()

        # Feature importance

        plt.figure(figsize = (16, 12))
        xgb.plot_importance(search.best_estimator_)
        

        plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "feature_importance.svg"))

        # Save Model
        filename = os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "model.sav")
        pickle.dump(search.best_estimator_, open(filename, 'wb'))

    # Get mean results and save in EVALUATION_PATH
        
    results["mean_accuracy"] = round(np.mean(results["accuracy"]), N_DECIMALS)
    results["mean_f1"] = round(np.mean(results["f1-score"]), N_DECIMALS)
        
    with open(os.path.join(EVALUATION_PATH, f'results.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results, f, ensure_ascii = False, indent = 4)

    
if __name__ == '__main__':

    args = parse_opt()

    main(args)