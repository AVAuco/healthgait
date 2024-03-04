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


from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectFromModel

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
    parser.add_argument('--grid_search', type = str, required = True, help = 'grid search json')
    parser.add_argument('--splits', type = int, required = True, help = 'number of split using in cross-validation')
    parser.add_argument('--njobs', type = int, required = True, help = 'number of jobs to run in parallel')
    parser.add_argument('--n_decimals', type = int, required = True, help = 'number of decimals to round results')
    parser.add_argument('--seed', type = int, default = 27, help = "seed")

    args = parser.parse_args()

    return args


def convert_floats(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError


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
        y = data[target].to_numpy(dtype='f')
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

    for partition_file in os.listdir(PARTITIONS_PATH):

        data = get_data(GAIT_PATH, MEASURES_PATH, os.path.join(PARTITIONS_PATH, partition_file), FEATURES, TARGET)

        data, target_scaler = preprocess_data(data)

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state = SEED, early_stopping_rounds = 100)

        cv = KFold(n_splits = SPLITS, random_state = SEED, shuffle = True)

        #search = GridSearchCV(model, param_grid, cv = cv, scoring='accuracy', n_jobs = NJOBS)

        search = BayesSearchCV(model, param_grid, cv = cv, scoring='neg_mean_squared_error', n_jobs = NJOBS, random_state = SEED)

        eval_set = [(data["val"]["data"], data["val"]["target"])]

        np.int = int
        #grid_search.fit(data["train"]["data"], data["train"]["target"], early_stopping_rounds = 10)
        search.fit(data["train"]["data"], data["train"]["target"], eval_set = eval_set, verbose = False)

        pathlib.Path(os.path.join(HYPERPARAMETERS_PATH, pathlib.Path(partition_file).stem)).mkdir(parents = True, exist_ok = True)

        search.best_params_["mse"] = search.best_score_

        # Save hyperparameters
        with open(os.path.join(HYPERPARAMETERS_PATH, pathlib.Path(partition_file).stem, f'best_params.json'), 'w', encoding = 'utf-8') as f:

            json.dump(search.best_params_, f, ensure_ascii = False, indent = 4)

        print(f"Grid Search completado para {partition_file}")

        # Model Evaluation

        y_preds = search.best_estimator_.predict(data["test"]["data"])

        y_preds = target_scaler.inverse_transform(y_preds.reshape(-1, 1))

        data["test"]["target"] = target_scaler.inverse_transform(data["test"]["target"].reshape(-1, 1))

        age_group_errors = evaluate_model_by_age_groups(data["test"]["target"], y_preds, age_groups)

        results_by_age["18-34"].append(age_group_errors["18-34"])
        results_by_age["35-49"].append(age_group_errors["35-49"]) 
        results_by_age["50-64"].append(age_group_errors["50-64"])


        # print(data["test"]["target"])

        # print(y_preds)

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

        # Feature importance

        # plt.figure(figsize = (16, 12))
        # xgb.plot_importance(search.best_estimator_)
        

        # plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "feature_importance.svg"))

        # plt.cla()

        # Relative error

        data = {"real_value": data["test"]["target"].flatten(), "predicted_value": y_preds.flatten()}

        df = pd.DataFrame(data)

        df["absolute_error"] = abs(df["real_value"] - df["predicted_value"])

        df["relative_error"] = (df["absolute_error"] / df["real_value"]) * 100

        # # Reemplazar infinitos con NaN
        # df["relative_error"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # # Opción 1: Eliminar filas con NaN (si se reemplazaron los infinitos con NaN)
        # df.dropna(subset=["relative_error"], inplace=True)

        df.sort_values(by=['relative_error'], ascending = False).to_csv(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "errors.csv"), index = False)

        n, bins, patches = plt.hist(x=df["relative_error"], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
        
        plt.yticks(fontsize = 15)
        plt.xticks(fontsize = 15)
        
        plt.xlabel('Relative error percentage', fontsize = 15)
        plt.ylabel('Frequency', fontsize = 15)
        plt.title(f'Relative error distribution for {TARGET.lower()} estimation', fontsize = 15)
        
        plt.savefig(os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "relative_error_distribution_weight.svg"))

        plt.cla()

        # Save Model
        filename = os.path.join(EVALUATION_PATH, pathlib.Path(partition_file).stem, "model.sav")
        pickle.dump(search.best_estimator_, open(filename, 'wb'))

    # Get mean results and save in EVALUATION_PATH
        
    results["mean_mse"] = np.mean(results["mse"])
    results["mean_mae"] = np.mean(results["mae"])
    results["mean_std_mae"] = round(np.mean(results["std_mae"]), N_DECIMALS)
        
    with open(os.path.join(EVALUATION_PATH, f'results.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results, f, ensure_ascii = False, indent = 4, default = convert_floats)


    results_by_age["18-34"] = np.mean(results_by_age["18-34"])
    results_by_age["35-49"] = np.mean(results_by_age["35-49"])
    results_by_age["50-64"] = np.mean(results_by_age["50-64"])

    with open(os.path.join(EVALUATION_PATH, f'age_group_errors.json'), 'w', encoding = 'utf-8') as f:

        json.dump(results_by_age, f, ensure_ascii = False, indent = 4, default = convert_floats)


    
if __name__ == '__main__':

    args = parse_opt()

    main(args)