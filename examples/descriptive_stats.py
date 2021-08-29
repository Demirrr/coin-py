import matplotlib.pyplot as plt
import coinpy as cp

# (1) Load dataframes holding low, high, open, close and volume of coins in 5 minutes interval
dfs = cp.DataFramesHolder(path='../Data')
# (2) Drop dataframes having length of less than 120 days
dfs.drop_data_frames(key=lambda x: len(x) < 12 * 24 * 120)
# (3) Create a feature based on the average value of open and close prices
dfs.create_feature(name='price', key=lambda df: (df['open'] + df['close']) / 2)
# (4) Create a feature represent real volume
dfs.create_feature(name='real_volume', key=lambda df: df['price'] * df['volume'])
dfs.create_feature(name='norm_price', key=lambda df: df['price'] / df['price'][0])
dfs.sort_frames(key=lambda coin_name_and_df: coin_name_and_df[1]['real_volume'].mean())
# [BTC ( 2021-02-20 10:55:00 -> 2021-08-28 15:20:00) : (52595, 8)]: low,	high,	open,	close,	volume,	price,	real_volume,	norm_price]
# [ETH ( 2021-02-20 10:55:00 -> 2021-08-28 15:20:00) : (52627, 8)]: low,	high,	open,	close,	volume,	price,	real_volume,	norm_price]
# [ADA ( 2021-03-18 16:05:00 -> 2021-08-28 15:20:00) : (44428, 8)]: low,	high,	open,	close,	volume,	price,	real_volume,	norm_price]
# . . .
dfs.select_col(['price'])
flag_compute_betas = True
flag_correlations = True

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

