import coinpy as cp
import numpy as np
import pandas as pd

# (1) Load all coins in
dfs = cp.DataFramesHolder(path='../Data')
# [ETC ( 2021-02-20 10:55:00 -> 2021-07-02 15:20:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# --
# [BTC ( 2021-02-20 10:55:00 -> 2021-07-02 15:15:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# (2) Create a new feature based on average of open and close
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
# (3) Select price
dfs.select_col(['price'])
# (4) Select coins
dfs.select_frames(['BTC', 'ADA', 'ETH', 'DOT', 'DOGE', 'XLM', 'UNI'])
# (5) Select Interval
dfs.select_interval(start="2021-07-1")
# (6) Normalize prices
dfs.normalize()
# (7) PLOT
dfs.plot(title='Returns', save='returns')
