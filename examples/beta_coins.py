import coinpy as cp
import numpy as np
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

betas = []
for c in df.columns:
    for cc in df.columns:
        if c == cc:
            continue
        df.plot(kind='scatter', x=c, y=cc, figsize=(12, 12))
        x, y = df.df[c].values, df.df[cc].values
        beta, alpha = np.polyfit(x, y, 1)[:]
        betas.append((beta, f'{cc} yielded Beta:{beta:.2f} x Alpha:{c}'))
        plt.plot(x, beta * x + alpha, '-', c='r')
        plt.title(f'Beta:{beta:.2f}, Alpha:{alpha:.2f}')
        plt.show()

betas.sort(key=lambda _: _[0], reverse=True)
for i in betas:
    print(i)
