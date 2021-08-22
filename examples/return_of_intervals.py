import matplotlib.pyplot as plt
import coinpy as cp
import numpy as np

dfs = cp.DataFramesHolder(path='../Data')
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.select_col(['price'])

compute_daily_returns = False
compute_hourly_returns = False
compute_10min_returns = True

if compute_daily_returns:
    for coin, df in dfs.average_returns(interval='D'):
        print(coin, df.describe())
if compute_hourly_returns:
    for coin, df in dfs.average_returns(interval='H'):
        print(coin, df.describe())
if compute_10min_returns:
    for c, df in dfs.average_returns(['ETH', 'BTC', 'DOT', 'ADA', 'DOGE'], interval='10T'):
        plt.plot(df, label=c)
        plt.legend()
        plt.title('Returns')
        plt.show()
