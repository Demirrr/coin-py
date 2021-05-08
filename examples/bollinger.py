import coinpy as cp
import matplotlib.pyplot as plt

dfs = cp.DataFramesHolder(path='../Data')
dfs.drop_data_frames(key=lambda x: len(x) < 1000)
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.select_col(['price'])
dfs.select_interval(start="2021-03-25")
# or dfs.select_interval(start="2021-03-25", end="2021-04-20")
for i in [1, 2, 3]:
    dfs.bollinger_bands('BTC', standard_variation=i, window_size=12 * 12)  # each window 5 minutes
    dfs.plot(coin='BTC')
