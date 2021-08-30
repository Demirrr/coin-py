import coinpy as cp
import numpy as np
import pandas as pd

# (1) Load all coins in
dfs = cp.DataFramesHolder(path='../Data')
# [ETC ( 2021-02-20 10:55:00 -> 2021-07-02 15:20:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# --
# [BTC ( 2021-02-20 10:55:00 -> 2021-07-02 15:15:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# (2) Select coins
dfs.select_frames(['BTC', 'ETH', 'ADA', 'DOGE', 'DOT', 'XLM', 'UNI'])
# (3) Select Interval
dfs.select_interval(start="2021-07-1")
# (4) Create a new feature based on normalized averaged of open and close prices
dfs.create_feature(name='price', key=lambda dataframe: cp.normalize((dataframe['open'] + dataframe['close']) / 2))
# (5) Select price
dfs.select_col(['price'])
# (6) PLOT
dfs.plot(title='Returns', save='returns')
