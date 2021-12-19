import coinpy as cp
import matplotlib.pyplot as plt

dfs = cp.DataFramesHolder(path='../Data')
dfs.drop_data_frames(key=lambda x: len(x) < 1000)
dfs.create_feature(name='price', key=lambda df: (df['open'] + df['close']) / 2)
dfs.select_col(['price'])
dfs.select_interval(start="2021-03-25", end="2021-07-20")

coins = ['BTC', 'ETH', 'ADA']

for c in coins:
    start_coin_budget = 0
    start_dolar_budget = 100_000

    data = dfs[c]

    window_size, standard_variation = 12 * 24, 3.0

    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * standard_variation)
    lower_band = rolling_mean - (rolling_std * standard_variation)

    decisions = []
    times = data.index

    price_higher_than_upper_bound = False
    price_lower_than_lower_bound = False

    for t in times:
        price = data.loc[t].values[0]

        # Buy or Sell ?
        upper_band_at_t = upper_band.loc[t].values[0]
        lower_band_at_t = lower_band.loc[t].values[0]

        if price > upper_band_at_t:
            # (1) Price Crossed the Upper Bound
            price_higher_than_upper_bound = True
        else:
            price_higher_than_upper_bound = False

        if price_higher_than_upper_bound:
            # (3) Check whether price return it again
            if price < upper_band_at_t:
                # Sell signal
                price_higher_than_upper_bound = False
                decisions.append(('SELL', t, price))

        if price < lower_band_at_t:
            # (4) Price Crossed the lower  Bound
            price_lower_than_lower_bound = True

        if price_lower_than_lower_bound:
            # Check whether price return it again
            if price > lower_band_at_t:
                # Sell signal
                price_lower_than_lower_bound = False
                decisions.append(('BUY', t, price))

    account = {'$': start_dolar_budget, c: start_coin_budget}
    for (d, time, price) in decisions:
        if d == 'SELL':
            # do we have BTC ?
            if account[c] > 0:
                account[c] -= 1
                account['$'] += price
            else:
                """ Cant do anything"""
        elif d == 'BUY':
            if account['$'] > price:
                account[c] += 1
                account['$'] -= price
            else:
                """ Cant do anything"""
        else:
            raise ValueError()

    print(f'Win in $ : {account["$"] - start_dolar_budget}')
    print(f'Win in {c} : {account[c] - start_coin_budget}')
