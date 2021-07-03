import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
dfs.select_col(['price'])
dfs.dropna()
#dfs.merge_frames(['ETH', 'BTC', 'ADA', 'XLM', 'UNI'])
dfs.select_frames(['BTC', 'ADA', 'ETH', 'XLM', 'UNI'])
dfs.select_interval(start="2021-03-25")#end="2021-04-20"
dfs.normalize()
dfs.portfolio_value(alloc=[.2, .2, .1, .3, .2])
dfs.plot()
