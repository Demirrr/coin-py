from .util import normalize_prices, get_all_folders, softmax
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from functools import reduce
from typing import List, Callable
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.style.use('seaborn-whitegrid')

np.random.seed(1)


class DataFrame:
    def __init__(self, df):
        self.df = df

    def __str__(self):
        return self.__repr__() + '\n' + str(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, col):
        return self.df[col]

    @property
    def index(self):
        return self.df.index

    @property
    def columns(self):
        return self.df.columns

    def normalize(self):
        self.df = normalize_prices(self.df)

    def plot(self, *args, **kwargs):
        self.df.plot(*args, **kwargs)

    def head(self, *args, **kwargs):
        return self.df.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.df.tail(*args, **kwargs)

    def compute_portfolio_value(self, allocation, money=1.0):
        self.df['PV'] = (self.df * allocation * money).sum(axis=1)

    def get_compute_portfolio_value(self, allocation, money=1.0):
        return (self.df * allocation * money).sum(axis=1)

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
        return averages.columns.tolist()

    def drop_rows_with_na(self):
        self.df.dropna(inplace=True)


class DataFramesHolder:
    def __init__(self, coins: List[str] = None, path: str = None):
        if coins is None:
            coins = []
            for i in get_all_folders(path):
                if 'csv' in i:
                    coins.append(i[i.rfind('/') + 1:-4])
        self.holder = self.read_csv(coins, path)
        # Test the order
        assert list(self.holder.keys()) == coins
        print('Drop frames having less than 10 value')
        self.drop_data_frames(key=lambda x: len(x) < 10)

    def __str__(self):
        m = ''
        for k, v in self.holder.items():
            columns = ',\t'.join([i for i in v.columns])
            start_time = v.index.min()
            end_time = v.index.max()
            m += f'\n[{k} ( {start_time} -> {end_time}) : {v.shape}]: {columns}]'
        return self.__repr__() + '\t' + m

    def __iter__(self):
        for k, v in self.holder.items():
            yield k, v

    def get_frame_names(self):
        return list(self.holder.keys())

    def __getitem__(self, item):
        return self.holder[item]

    @property
    def values(self):
        assert len(self.holder) == 1
        _, v = self.holder.popitem()
        return v

    def normalize(self):
        for k, v in self.holder.items():
            self.holder[k] = v / v.iloc[0]

    def find(self, n=3, key=None, descending=True):
        top = []
        for k, v in self.holder.items():
            top.append((k, key(v.values)))
        top.sort(key=lambda tup: tup[1], reverse=descending)
        return [top[_] for _ in range(n)]

    def dropna(self):
        for k, v in self.holder.items():
            self.holder[k].dropna(inplace=True)

    def drop_data_frames(self, key: Callable) -> None:
        """
        Drop dataframes not satisfying the constraint.
        :param key:
        :return:
        """
        coins = list(self.holder.keys())
        for i in coins:
            if key(self.holder[i]):
                del self.holder[i]

    def sort_frames(self, key=lambda kdf: kdf[1].volume.mean(), reverse=True):
        self.holder = OrderedDict(sorted(self.holder.items(), key=key, reverse=reverse))

    def select_col(self, cols: List):
        coins = list(self.holder.keys())
        for i in coins:
            self.holder[i] = self.holder[i][cols]

    def __average_interval_return(self, *, coin: str, interval: str) -> np.float64:
        """
        Normalize prices and average them

        :param coin: cryptocurrency
        :param interval: Resample rule  pandas.DataFrame.resample.html#pandas-dataframe-resample
        :return:
        """
        # (1) Normalize prices
        normalized_prices = ( self[coin] / self[coin].iloc[0] ) - 1
        # (2) Average results
        averaged_normalized_prices = normalized_prices.resample(interval).mean()
        return averaged_normalized_prices

    def average_returns(self, coin=None, interval='D'):
        """
        :param coin:
        :param interval: Downsample with Day (D), X Days (XD), X Minutes (XT)
        :return:
        """
        if isinstance(coin, str):
            return self.__average_interval_return(coin=coin, interval=interval)
        elif isinstance(coin, list):
            for k in coin:
                yield k, self.__average_interval_return(coin=k, interval=interval)
        elif coin is None:
            for (k, v) in self:
                yield k, self.__average_interval_return(coin=k, interval=interval)
        else:
            raise KeyError(f'coin={coin} is not valid')

    @staticmethod
    def compute_return_df(df, interval='D') -> pd.DataFrame:
        """
        Compute return on time series data (df).
        :param df:
        :param interval:
        :return:
        """
        # 1. Average values in a given interval
        return_in_interval = df.resample(interval).mean()
        # 2. Compute returns.
        return_in_interval[1:] = (return_in_interval[1:] / return_in_interval[:-1].values) - 1
        # 3. Set 0 to first row
        return_in_interval.iloc[0, :] = 0  # set daily return for row 0 to 0.

        return_in_interval.dropna(inplace=True)
        return return_in_interval

    def __compute_beta(self, dfx: pd.DataFrame, dfy: pd.DataFrame, interval: str):
        # 1. Compute Returns
        return_intervals = pd.merge(self.compute_return_df(dfx, interval), self.compute_return_df(dfy, interval),
                                    on='time', how='inner')

        start_time = return_intervals.index.min()
        end_time = return_intervals.index.max()

        return_intervals = return_intervals.values
        return_intervals_dfx, return_intervals_dfy = return_intervals[:, 0], return_intervals[:, 1]
        beta, alpha = np.polyfit(x=return_intervals_dfx, y=return_intervals_dfy, deg=1)[:]
        return beta, alpha, start_time, end_time, return_intervals

    def __beta_single(self, x=None, y=None, plot=True, interval='D'):
        beta, alpha, start_time, end_time, returns = self.__compute_beta(self[x], self[y], interval)
        if alpha > 0:
            info = f'{beta:.2f} {x} + {alpha:.2f}={y}'
        else:
            info = f'{beta:.2f} {x} {alpha:.2f}={y}'
        if plot:
            x_return, y_return = returns[:, 0], returns[:, 1]
            plt.scatter(x_return, y_return, s=1.0)
            plt.plot(x_return, beta * x_return + alpha, '-', c='r')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(info)
            plt.show()

        return beta, info

    def beta(self, x: str = None, y: str = None, plot: bool = True, interval: str = 'D'):
        # Compute BETA for all coins
        if x is None and y is None:
            betas = []
            for (x, dfx) in self:
                for (y, dfy) in self:
                    if x == y:
                        continue
                    result = self.__beta_single(x, y, plot, interval)
                    betas.append(result)
            betas.sort(key=lambda _: _[0], reverse=True)
            return betas
        else:
            beta, info = self.__beta_single(x, y, plot, interval)
            print(info)
            return beta

    def corr(self, x: str = None, y: str = None):
        # Compute pairwise Pearson correlation coefficient: P(X,Y) = Covariance(X,Y) / (std_X * std_Y)
        if x is None and y is None:
            corr = []
            seen = []
            for (x, dfx) in self:
                for (y, dfy) in self:
                    if x == y or (x + ':' + y in seen) or (y + ':' + x in seen):
                        continue

                    df_xy = pd.merge(self[x], self[y], on='time', how='inner')
                    seen.append(x + ':' + y)
                    seen.append(y + ':' + x)
                    c = df_xy.iloc[:, 0].corr(df_xy.iloc[:, 1])
                    corr.append((c, f'{x} : {y}: {c:.3f}'))

            corr.sort(key=lambda _: _[0], reverse=True)
            return corr
        else:
            df_xy = pd.merge(self[x], self[y], on='time', how='inner')
            corr = df_xy.iloc[:, 0].corr(df_xy.iloc[:, 1])
            print(f'{x} : {y}: {corr:.3f}')

    @staticmethod
    def read_csv(coins, path):
        results = dict()
        for c in coins:
            df = pd.read_csv(path + '/' + c + '.csv', index_col='time', parse_dates=True)
            if not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)
            results[c] = df

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

    def select_with(self, index):
        keys_to_del = list(self.holder.keys())
        for i in keys_to_del:
            if len(index) != len(self.holder[i]):
                del self.holder[i]

    def select_frames(self, coins):
        keys_to_del = self.holder.keys() - coins
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

    def merge_frames(self, frames, how='inner'):

        df = []
        names = ''
        for i in frames:
            df.append(self.holder[i])
            names += '_' + i

        df = reduce(lambda left, right: pd.merge(left, right, on='time', how=how), df)
        df.columns = frames
        self.holder.clear()
        self.holder[names] = df

    def select_interval(self, start: str, end: str = None) -> None:
        """
        Given start and end, select only those dataframes that has values for such interval.
        """
        try:
            assert isinstance(start, str)
        except AssertionError:
            print(f'Incorrect input param start: {start}')
            exit(1)

        keys_to_del = []
        for k, v in self.holder.items():
            if start in v.index:
                if end is None:
                    self.holder[k] = v.loc[start:]
                else:
                    self.holder[k] = v[start:end]

            else:
                keys_to_del.append(k)
        for k in keys_to_del:
            del self.holder[k]

    def select_last(self, n=12 * 24 * 7):
        for k, v in self.holder.items():
            self.holder[k] = v.tail(n)

    def plot(self, coin=None, title='', start=None, save=None):
        """
        PLOT COINTS
        :param save:
        :param coin:
        :param title:
        :param start:
        :return:
        """
        if coin is None:
            for k, v in self.holder.items():
                if start:
                    v = self[k].loc[start]
                plt.plot(v, label=k)
                print(k)
        else:
            assert isinstance(coin, list)
            for c in coin:
                if start:
                    cm = self[c].loc[start]
                else:
                    cm = self[c]

                for column in cm:
                    plt.plot(cm[column], label=column)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(save + '.png')
        plt.show()

    def portfolio_value(self, coin_name=list, alloc=list) -> None:
        pv = None
        for cn, alloc_val in zip(coin_name, alloc):
            if pv is None:
                pv = self.holder[cn] * alloc_val
            else:
                pv = pv.merge(self.holder[cn] * alloc_val, left_index=True, right_index=True)

        self.holder['PV'] = pv.sum(axis=1)

    def optim_portfolio_value(self, n=10, money=1, method='SLSQP', plot=True) -> None:
        """
        Optimize portfolio value by random sampling.
        :param method:
        :param n:
        :param money:
        :param plot:
        :return:
        """

        def f(x):
            return (df * x).sum(axis=1)[-1]

        def neg_f(x):
            return -(df * x).sum(axis=1)[-1]

        res = []
        # 1. Compute portfolio values with randomly sampled allocations.
        for coin_name, df in self:
            _, d = df.shape
            alloc = None
            last_pv = None

            if method == 'Random':
                X = softmax(np.random.randn(n * d).reshape((n, d)))
                X *= money
                for alloc in X:
                    last_pv = f(alloc)
            elif method == 'SLSQP':
                # 1. Initial value
                X_init = softmax(np.random.randn(d).reshape((1, d)))[0]
                # 2. Constraint
                constraints = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
                # 3. Bounds => Required to have non negative values, min-max values
                bounds = tuple((0.0, 1.0) for _ in X_init)
                min_result = minimize(neg_f, X_init, method='SLSQP', bounds=bounds,
                                      constraints=constraints,
                                      options={'maxiter': n})
                alloc = np.around(min_result.x, 3)
                last_pv = f(alloc)
            else:
                raise ValueError()
            res.append((last_pv, coin_name, dict(zip(df.columns.tolist(), alloc)), len(df)))

        # 2. Sort in descending order of portfolio values.
        res.sort(key=lambda x: x[0], reverse=True)
        best = res[0]
        # 3. Show the best result
        print(f'PV :{best[0]} of allocation:{best[2]}')

        # 4. Apply best found allocation
        # 4.1. Get the name of frame
        coin_name = best[1]
        # 4.2. Get coin allocation mapping
        cols = list(best[2].keys())
        best_alloc = list(best[2].values())
        # 4.3 Sanity checking
        assert self[coin_name].columns.tolist() == cols
        self.portfolio_value(coin_name, best_alloc)
        # 5. Plot it.
        if plot:
            self.plot()

    def bollinger_bands(self, data_frame, window_size=50, standard_variation=2.0):
        rolling_mean = self[data_frame].rolling(window=window_size).mean()
        rolling_std = self[data_frame].rolling(window=window_size).std()

        upper_band = rolling_mean + (rolling_std * standard_variation)
        lower_band = rolling_mean - (rolling_std * standard_variation)

        # self[data_frame]['rolling_mean']=rolling_mean
        self[data_frame]['upper_band'] = upper_band
        self[data_frame]['lower_band'] = lower_band

        # return rolling_mean, (upper_band, lower_band)

    def apply_bollinger_bands_on_normalized_values_given_time(self, dataframe,
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

    def create_time_table(self):
        df = None
        for k, v in self.holder.items():
            if df is None:
                df = pd.DataFrame(data=v.values, index=v.index, columns=[k])
            else:
                df = pd.merge(df, pd.DataFrame(data=v.values, index=v.index, columns=[k]), on='time', how='outer')
        return df
