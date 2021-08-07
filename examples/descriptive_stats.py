import matplotlib.pyplot as plt

import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.drop_data_frames(key=lambda x: len(x) < 5000)
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.sort_frames()
dfs.select_col(['price'])
dfs.dropna()

flag_compute_betas = False
flag_correlations = False
flag_compute_returns = True


def compute_betas():
    """
    Compute Betas on daily returns
    :return:
    """
    dfs.beta(x='XLM', y='UNI', plot=True)
    dfs.beta(x='XLM', y='ETH', plot=True)
    dfs.beta(x='ETH', y='UNI', plot=True)
    dfs.beta(x='BTC', y='UNI', plot=True)
    dfs.beta(x='BTC', y='ETH', plot=True)

    # To compute all betas
    betas = dfs.beta(interval='D', plot=False)
    for (b, i) in betas[:10]:
        print(i)


if flag_compute_betas:
    compute_betas()

if flag_correlations:
    dfs.corr(x='XLM', y='UNI')
    dfs.corr(x='XLM', y='ETH')
    dfs.corr(x='XLM', y='BTC')
    dfs.corr(x='ETH', y='UNI')
    dfs.corr(x='BTC', y='UNI')
    dfs.corr(x='BTC', y='ETH')

    corr = dfs.corr()
    print('\n\nTop 10 positive correlation')
    for i, (c, info) in enumerate(corr[:10]):
        print(f'{i + 1}. {info}')

    print('\n\nTop 10 negative correlation')
    for i, (c, info) in enumerate(corr[-10:]):
        print(f'{i + 1}. {info}')

if flag_compute_returns:
    # Normalize prices
    dfs.normalize()
    print('Total return: ', dfs.find(n=10, key=lambda price: price[-1] - price[0], descending=True))
    print('Average return:', dfs.find(n=10, key=lambda price: price.mean(), descending=True))
    print('Standard deviation return: ', dfs.find(n=10, key=lambda price: price.std(), descending=True))
