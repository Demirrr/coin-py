import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.select_col(['price'])
dfs.dropna()
dfs.merge_frames(['ETH', 'BTC', 'ADA', 'XLM', 'UNI'])
dfs.select_interval(start="2021-04-15", end="2021-05-15")
dfs.normalize()

dfs.optim_portfolio_value(n=10, method='SLSQP')
dfs.optim_portfolio_value(n=10, method='Random')
