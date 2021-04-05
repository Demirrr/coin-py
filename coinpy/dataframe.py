from .util import normalize_prices, get_all_folders
import pandas as pd
from functools import reduce
from typing import List
import matplotlib.pyplot as plt


class DataFrame:
    def __init__(self, df):
        self.df = df

    def __str__(self):
        return self.__repr__() + '\n' + str(self.df)

    def normalize(self):
        self.df = normalize_prices(self.df)

    def plot(self, *args, **kwargs):
        self.df.plot(*args, **kwargs)

    def head(self, *args, **kwargs):
        return self.df.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.df.tail(*args, **kwargs)

    @property
    def columns(self):
        return self.df.columns

    def compute_portfolio_value(self, allocation, money=1.0):
        self.df['PV'] = (self.df * allocation * money).sum(axis=1)

    def compute_moving(self, windows, apply):
        result = []
        for i in windows:
            if apply == 'mean':
                m = self.df.rolling(window=i).mean()
                m.rename(columns=dict(zip(m.columns, [apply + '_Of_' + str(i) + '_' + str(_) for _ in m.columns])),
                         inplace=True)
                result.append(m)
            elif apply == 'std':
                raise ValueError
            else:
                raise KeyError

        averages = pd.concat(result, axis=1)
        assert (averages.index == self.df.index).all()
        n, _ = self.df.shape
        self.df = pd.concat([self.df, averages], axis=1)
        assert n == len(self.df)

    def drop_rows_with_na(self):
        self.df.dropna(inplace=True)


class DataFramesHolder:
    def __init__(self, coins: List[str] = None, path: str = None):
        if coins is None:
            coins = []
            for i in get_all_folders(path):
                coins.append(i[i.rfind('/') + 1:-4])
        self.holder = self.read_csv(coins, path)

        # Test the order
        assert list(self.holder.keys()) == coins

    def __str__(self):
        m = ''
        for k, v in self.holder.items():
            columns = ',\t'.join([i for i in v.columns])
            m += f'\n[{k}:{v.shape}:Columns:{columns}]'
        return self.__repr__() + '\t' + m

    @staticmethod
    def read_csv(coins, path):
        results = dict()
        for c in coins:
            results[c] = pd.read_csv(path + '/' + c + '.csv', index_col='time', parse_dates=True)
        return results

    def rename_coins(self, names):
        self.holder = dict(zip(names, self.holder.values()))

    def rename_columns(self, names):
        for k, v in self.holder.items():
            v.rename(columns=dict(zip(v.columns, names)), inplace=True)

    def preprocess(self, mapping):
        for k, v in self.holder.items():
            if mapping['func'] == 'mean':
                v[mapping['output']] = v[mapping['input']].mean(axis=1)
            else:
                raise ValueError
            self.holder[k] = v

    def pipeline(self, steps):
        for i in steps:
            for k, v in i.items():
                func = getattr(self, k, None)
                assert callable(func)
                res = func(v)
                if res is not None:
                    return res

    def select_frames(self, m):
        keys_to_del = self.holder.keys() - m
        for i in keys_to_del:
            del self.holder[i]

    def select(self, m):
        frames = []
        for k, v in self.holder.items():
            df = v[m].copy()
            df.rename(columns=dict(zip(df.columns, [k + '_' + i for i in m])), inplace=True)
            frames.append(df)

        # Sort frames by the length of the time interval
        frames.sort(key=lambda x: len(x), reverse=True)

        df = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), frames)
        ######################################################################
        # Method    SQL JOIN NAME       Description
        ######################################################################
        # left      LEFT OUTER JOIN     Use keys from left frame only .
        # right     RIGHT OUTER JOIN    Use keys from right frame only .
        # outer     FULL OUTER JOIN     Use union of keys from both frames .
        # inner     INNER JOIN          Use intersection of keys from frames .
        ######################################################################
        del self.holder
        return DataFrame(df)