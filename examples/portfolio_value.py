import coinpy as cp

dfs = cp.DataFramesHolder(path='../Data')
dfs.select_frames(['BTC', 'ADA', 'ETH', 'XLM', 'UNI'])
dfs.select_interval(start="2021-03-25", end="2021-08-01")
dfs.create_feature(name='price', key=lambda dataframe: cp.normalize((dataframe['open'] + dataframe['close']) / 2))
dfs.select_col(['price'])
dfs.portfolio_value(coin_name=['BTC', 'ETH', 'ADA', 'XLM'], alloc=[.1, .6, .1, .1, .1])
dfs.plot(title='Portfolio value')
