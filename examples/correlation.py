import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.drop_data_frames(key=lambda x: len(x) < 1000)
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.sort_frames()
dfs.select_col(['price'])
dfs.dropna()

dfs.corr(x='XLM', y='UNI')
dfs.corr(x='XLM', y='ETH')
dfs.corr(x='XLM', y='BTC')
dfs.corr(x='ETH', y='UNI')
dfs.corr(x='BTC', y='UNI')
dfs.corr(x='BTC', y='ETH')

corr = dfs.corr()
print('\n\nTop 10 positive correlation')
for i, (c, info) in enumerate(corr[:10]):
    print(f'{i+1}. {info}')

print('\n\nTop 10 negative correlation')
for i, (c, info) in enumerate(corr[-10:]):
    print(f'{i+1}. {info}')
