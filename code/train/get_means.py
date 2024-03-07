import os
import sys
import json
import pathlib
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def parse_opt():

    parser = argparse.ArgumentParser(
        description = 'Calculate mean evaluation results'
    )

    parser.add_argument('--results_path', type = str, required = True, help = 'root directory')
    parser.add_argument('--n_decimals', type = int, required = True, help = 'number of decimals to round results')

    args = parser.parse_args()

    return args

def main(args):
   
    ############################################################################
    ######################### COMMAND LINE ARGUMENTS ###########################
    ############################################################################

    RESULTS_PATH = args.results_path
    N_DECIMALS = args.n_decimals


    ############################################################################
    ############################# INICIALIZATION ###############################
    ############################################################################

    data = []

    for folder_result in os.listdir(RESULTS_PATH):

        if os.path.join(RESULTS_PATH, folder_result).isdir():

            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_result, 'partition_results.csv'), index_col=None, header=0)
            data.append(df)

    df = pd.concat(data, axis=0, ignore_index=True)


    ############################################################################
    ############################# CALCULATE MEAN ###############################
    ############################################################################

    metrics = {"mae": round(df["mae"].mean(), N_DECIMALS),
               "std_mae": round(df["std_mae"].mean(), N_DECIMALS)
    }


    results = pd.DataFrame(data = [list(metrics.values())], columns = list(metrics.keys()))

    # Almacenar resultados fichero csv

    results.to_csv(os.path.join(RESULTS_PATH, "results.csv"), index = False)


    


if __name__ == "__main__":

    args = parse_opt()

    main(args)