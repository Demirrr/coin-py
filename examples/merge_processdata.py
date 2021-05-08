"""
Merge crypto data stored in individual csv
"""
import pandas as pd

from coinpy.util import get_all_folders, get_all_data, load_csv
import argparse
import os


def get_df_to_be_inserted(path):
    for i in get_all_folders(path):
        df_to_insert, _ = load_csv(i)
        coin_csv_name = i[i.rfind('/') + 1:]
        yield df_to_insert, coin_csv_name


def get_main_df(path_main, coin_csv):
    path_main = path_main + '/' + coin_csv
    df_main, _ = load_csv(path_main)
    try:
        assert _
    except AssertionError:
        print(f'{path_main} could not found. ')
        df_main = pd.DataFrame()
    return df_main


def preprocessing(args):
    for df_to_insert, coin_csv_name in get_df_to_be_inserted(args.path_data_to_insert):
        df_main = get_main_df(args.path_main_data, coin_csv_name)

        try:
            df_main = df_main.append(df_to_insert, verify_integrity=True)
        except ValueError:
            print(f'{coin_csv_name} has previously overlapping values')
            df_main = df_main.append(df_to_insert, verify_integrity=False)
            df_main.drop_duplicates(inplace=True)

        try:
            os.mkdir(args.path_merged_data)
        except FileExistsError:
            """ File exists do not need to create a folder """

        df_main.to_csv(args.path_merged_data + '/' + coin_csv_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data_to_insert", default='../ProcessedData')
    parser.add_argument("--path_main_data", default='../Data')
    parser.add_argument("--path_merged_data", default='../Data')

    preprocessing(parser.parse_args())
