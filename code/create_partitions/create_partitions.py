import os
import sys
import json
import pathlib
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def parse_opt():

    parser = argparse.ArgumentParser(description='Create training partitions for the UCA GAIT dataset training')
    parser.add_argument('--validation_ratio', type=float, required=True)
    parser.add_argument('--patients_measures', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--k_fold', type=int, default = 4)
    parser.add_argument('--variables', nargs='+', required=True)
    parser.add_argument('--age_range', type=str, choices=['young', 'adult', 'old'], required=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()

def preprocess_data(df, variables, age_range = None):


    selected_columns = [var for var in variables if var in df.columns]
    selected_columns.append("ID")

    df = df[selected_columns]

    df = df.dropna()
    df = df.reset_index(drop = True)

    if 'Age' in variables:

        df.loc[:, 'Age'] = df['Age'].apply(lambda x: categorize_age(x))

    if 'BMI' in variables:

        df.loc[:, 'BMI'] = df['BMI'].apply(lambda x: categorize_bmi(x))

    if age_range:
        
        df = df[df['Age'] == age_range]

    if 'Age' in variables and 'Age' not in selected_columns:
        selected_columns.append('Age')

    # print(df)
   
    return df

def categorize_age(age):

    if 18 <= age <= 34:

        return "young"
    
    elif 35 <= age <= 49:

        return "adult"
    
    elif 50 <= age <= 64:

        return "old"
    
def categorize_bmi(bmi):

    if bmi < 18.5:

        return "thinness"

    elif bmi >= 18.5 and bmi <= 24.9:

        return "normal"

    elif bmi >= 25.0 and bmi <=29.9:

        return "overweight"

    else:

        return "obese"

def check_variables(df, variables):

    for var in variables:

        if var not in df.columns:

            sys.exit(f"Error: '{var}' not found in the DataFrame.")

def create_stratification_tag(df, variables):

    return df[variables].astype(str).agg('_'.join, axis=1)

def save_partitions(output_path, partitions, file_name):

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(output_path, file_name), 'w') as f:

        json.dump(partitions, f, ensure_ascii=False, indent=4)

def show_partitions(df, train_patients, validation_patients, test_patients):

    print(f"Train: {train_patients.shape[0]}, Validation: {validation_patients.shape[0]}, Test: {test_patients.shape[0]}")

    print("\nClass Distribution Train:")
    print(df[df['ID'].isin(train_patients)]["tag"].value_counts())
    print("\nClass Distribution Validation:")
    print(df[df['ID'].isin(validation_patients)]["tag"].value_counts())
    print("\nClass Distribution Test:")
    print(df[df['ID'].isin(test_patients)]["tag"].value_counts())

    
def main(args):

    df = pd.read_csv(args.patients_measures)
    df = preprocess_data(df, args.variables, args.age_range)

    check_variables(df, args.variables)
    df['tag'] = create_stratification_tag(df, args.variables)

    n_patients = df["ID"].shape[0]

    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)

    for i, (train_index, test_index) in enumerate(skf.split(df["ID"], df['tag'])):

        train_ratio = train_index.shape[0] / n_patients


        test_patients = df["ID"][test_index]

        # Obtener el conjunto de validación
        train_patients, validation_patients = train_test_split(df["ID"][train_index], 
                    train_size = train_ratio / (train_ratio + args.validation_ratio), 
                    stratify = df[df['ID'].isin(df["ID"][train_index])]["tag"],
                    random_state = args.seed)
        
        if args.verbose:

            show_partitions(df, train_patients, validation_patients, test_patients)

        save_partitions(args.output_path, {
            "train": sorted(list(train_patients.values)),
            "validation": sorted(list(validation_patients.values)),
            "test": sorted(list(test_patients.values))
        }, f"partition_{i}.json")



if __name__ == "__main__":

    args = parse_opt()
    main(args)
