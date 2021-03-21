import os
import datetime
from typing import List, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def create_experiment_folder(folder_name):
    if folder_name is None:
        folder_name = 'Data'
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
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
        except AssertionError:
            print(p, 'will be ignored')
            continue
        df_collect = df_collect.join(df_temp)

    return df_collect


def data_per_days_collector(data_days_path: List[str], cryptos: List):
    """

    :param data_days_path:
    :param cryptos:
    :return:
    """

    df = data_per_day_collector(data_days_path[0], cryptos)
    for i in data_days_path[1:]:
        df_temp = data_per_day_collector(i, cryptos)
        df = df.append(df_temp)

    df = df.sort_values(by=['time'])
    return df


def bollinger_bands(data_frame, window_size=10, standard_variation=2.0):
    rolling_mean = data_frame.rolling(window=window_size).mean()
    rolling_std = data_frame.rolling(window=window_size).std()

    upper_band = rolling_mean + (rolling_std * standard_variation)
    lower_band = rolling_mean - (rolling_std * standard_variation)

    return rolling_mean, (upper_band, lower_band)


def apply_bollinger_bands_on_normalized_values_given_time(dataframe,
                                                          features=None,
                                                          window_size=5,
                                                          min_range=None,
                                                          max_range=None):
    assert features
    if min_range is None and max_range is None:
        min_range, max_range = dataframe.index.min(), dataframe.index.max()

    dataframe = specific_time_range(dataframe, min_range=min_range, max_range=max_range)[features]

    normalized_df = normalize_df(dataframe)
    plot(normalized_df, title='Normalized prices in a given interval')

    # 5 minutes 5 => 25 minutes
    rolling_means, bands = bollinger_bands(normalized_df, window_size=window_size)
    upper, lower = bands
    for i in features:
        normalized_df[i].plot(label=i)
        upper[i].plot(label='Upper_Bound')
        lower[i].plot(label='Lower_Bound')

        plt.title('Bollinger bands')
        plt.legend()
        plt.show()


def deprecated_compute_returns_from_scratch(data_frame, by=1):
    """

    :param data_frame:
    :param by: indicates every 5 x by minutes
    :return:
    """

    df_returns = data_frame.copy()
    df_returns[by:] = (data_frame[by:] / data_frame[:-by].values) - 1
    df_returns.iloc[0:by, :] = 0  # set daily returns for row 0 to 0
    return df_returns


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
        data_frame.drop(columns=['low_' + i, 'high_' + i, 'open_' + i, 'close_' + i, 'volume_' + i], inplace=True)
    return data_frame


def get_all_data(cryptos):
    days = ['Data/' + i + '/' for i in os.listdir('Data')]
    df = data_per_days_collector(data_days_path=days,
                                 cryptos=cryptos)
    return df


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
