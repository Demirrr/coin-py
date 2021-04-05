import coinpy as cp
import matplotlib.pyplot as plt

coins = ['BTC-USD', 'ETH-USD', 'UNI-USD']
dfs = cp.DataFramesHolder(coins=coins, path='../ProcessedData')

df = dfs.pipeline([
    {'rename_coins': ['BTC', 'ETH', 'UNI']},
    {'rename_columns': ['low', 'high', 'open', 'close', 'volume']},
    {'preprocess': {'func': 'mean', 'input': ['open', 'close'], 'output': 'price'}},
    {'select_frames': ['BTC', 'ETH', 'UNI']},
    {'select': ['price']}

])
del dfs
df.normalize()
df.drop_rows_with_na()
df.plot()
plt.legend()
plt.show()
