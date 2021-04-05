from .util import normalize_prices, get_all_folders
import pandas as pd
from functools import reduce
from typing import List
import matplotlib.pyplot as plt


class DataFrames:
    def __init__(self, coins: List[str] = None, path: str = None):
        if coins is None:
            coins = []
            for i in get_all_folders(path):
                coins.append(i[i.rfind('/') + 1:-4])

        self.holder = self.read_csv(coins, path)
        self.dataframe = None

    def __str__(self):
        return self.__repr__()+'\n'+str(self.dataframe)

    @staticmethod
    def read_csv(coins, path):
        results = dict()
        for c in coins:
            results[c] = pd.read_csv(path + '/' + c + '.csv', index_col='time', parse_dates=True)
        return results


    def normalize(self):
        self.dataframe = normalize_prices(self.dataframe)

    def plot(self, title=''):
        self.dataframe.plot(title=title)
        plt.legend()
        plt.show()

    def head(self, *args, **kwargs):
        return self.dataframe.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.dataframe.tail(*args, **kwargs)

    def compute_portfolio_value(self, allocation, money=1.0):
        self.dataframe['PV'] = (self.dataframe * allocation * money).sum(axis=1)
        return self.dataframe

    def columns(self):
        return self.dataframe.columns

    def compute_moving(self, windows, apply):
        d = pd.DataFrame()
        for i in windows:
            if apply == 'mean':
                d[apply + '_Of_' + str(i)] = self.dataframe.rolling(window=i).mean()
            elif apply == 'std':
                d[apply + '_Of_' + str(i)] = self.dataframe.rolling(window=i).std()
            else:
                raise KeyError
        self.dataframe = pd.concat([self.dataframe, d], axis=1)