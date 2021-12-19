import os
import datetime
from typing import List, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

pd.options.display.max_columns = None


def specific_time_range(data_frame, min_range, max_range):
    """
    :param max_range:
    :param min_range:
    :param data_frame:
    :return:
    """
    return data_frame[min_range:max_range]


def normalize_df(data_frame):
    """
    Normalize time-seried by interval with the starting value
    :paramd ata_frame:
    :return:
    """
    return data_frame / data_frame.iloc[0, :]


def plot(data_frame, title='Default'):
    data_frame.plot(title=title)
    plt.tight_layout()
    plt.show()


def create_experiment_folder(directory):
    if directory is None:
        directory = os.getcwd() + '/' + 'Fetched'
    else:
        directory += '/' + 'Fetched'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + '/' + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder


def data_per_day_collector(data_day_path: str, cryptos: List):
    """
    Each CSV file ordered as descending order of time.
    1. Create first dataframe *df_collect* to store data
    2. Iteratively fill df_collect
    2.1 Creat the next data
    2.2 Store it
    2.3. Check the index, i.e. time interval. Time must match.
    :param data_day_path: A path leading to folder
    :param cryptos: A list of strings indicating product
    :return: DataFrame
    """
    # 1.
    df_collect = pd.read_csv(data_day_path + cryptos[0] + '.csv', index_col='time', parse_dates=True)

    for c in cryptos[1:]:
        p = data_day_path + c + '.csv'
        df_temp = pd.read_csv(p, index_col='time', parse_dates=True)
        try:
            # Time intervals must be same
            assert (df_temp.index == df_collect.index).all()
            df_collect = df_collect.join(df_temp)
        except:
            print(p, 'will be ignored')

    return df_collect


def load_csv(path):
    """
    If a csv file exist read it and return true
    otherwise return tuple of Nones
    :param path:
    :return:
    """
    try:
        return pd.read_csv(path, index_col='time', parse_dates=True), True
    except FileNotFoundError:
        # print(f'File not Found => {path}')
        return None, None


def collect_dataframes(paths: List[str], data: str):
    """

    :param paths:
    :param data:
    :return:
    """

    results = (load_csv(path=i + '/' + data + '.csv') for i in paths)
    dfs = (df for df, flag in results if flag)
    df = next(dfs)
    df.columns = df.columns.str.replace('_' + data, '')
    for i in dfs:
        # Remove the Tag from columns e.g. _BTC-USD.
        i.columns = i.columns.str.replace('_' + data, '')
        df = df.append(i)

    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_values(by=['time'])
    return df



def normalize_prices(data_frame):
    """

    :param data_frame:
    :return:
    """
    return data_frame / data_frame.iloc[0]


def compute_returns(data_frame, by=1):
    """

    :param data_frame:
    :param by: indicates every 5 x by minutes
    :return:
    """
    assert by > 0

    if isinstance(data_frame, pd.DataFrame):
        df_returns: Union[int, Any] = (data_frame / data_frame.shift(by)) - 1
        df_returns.iloc[0:by, :] = 0  # set daily returns for row 0 to 0
        return df_returns
    else:
        df_returns: Union[int, Any] = (data_frame / data_frame.shift(by)) - 1
        return df_returns[by:]


def compute_weighted_price(data_frame, c):
    # Compute Price by averaging opening and closing price in 5 minutes interval.
    for i in c:
        data_frame[i] = (data_frame['open_' + i] + data_frame['close_' + i]) / 2.0
        data_frame.drop_data_frames(columns=['low_' + i, 'high_' + i, 'open_' + i, 'close_' + i, 'volume_' + i],
                                    inplace=True)
    return data_frame


def get_all_folders(path):
    """

    :param path:
    :return:
    """
    return [path + '/' + i for i in os.listdir(path)]


def get_all_data(data, path):
    return collect_dataframes(paths=get_all_folders(path), data=data)


def compute_allocation(df, allocs):
    df *= allocs
    return df


def compute_pv(df, allocs, money):
    data_frame = df.copy()
    data_frame = compute_allocation(data_frame, allocs=allocs)
    data_frame *= money
    data_frame['PV'] = data_frame.sum(axis=1)
    return data_frame


def softmax(s):
    s -= np.max(s, axis=1, keepdims=True)
    exp_scores = np.exp(s)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def load_json(p):
    with open(p, 'r') as file_descriptor:
        return json.load(file_descriptor)
