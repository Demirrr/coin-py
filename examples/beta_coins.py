import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.drop_data_frames(key=lambda x: len(x) < 1000)
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.sort_frames()
dfs.select_col(['price'])
dfs.dropna()

dfs.beta(x='XLM', y='UNI', plot=True)
dfs.beta(x='XLM', y='ETH', plot=True)
dfs.beta(x='ETH', y='UNI', plot=True)
dfs.beta(x='BTC', y='UNI', plot=True)
dfs.beta(x='BTC', y='ETH', plot=True)

# To compute all betas
# betas = dfs.beta(interval='D', plot=False)
# for (b, info) in betas[:10]:
#    print(info)
