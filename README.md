# CoinPy
CoinPy is an open-source project for cryptocurrency data acquisition and modelling.

## Installation
```bash
pip install coin-py
```
Installation from source
```bash
git clone https://github.com/Demirrr/coin-py
# To develop
python pip install -e
python -m pytest tests
```
### Examples
```python
# (1) Load all coins in
dfs = cp.DataFramesHolder(path='../Data')
# [ETC ( 2021-02-20 10:55:00 -> 2021-07-02 15:20:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# --
# [BTC ( 2021-02-20 10:55:00 -> 2021-07-02 15:15:00) : (36174, 5)]: low,	high,	open,	close,	volume]
# (2) Select coins
dfs.select_frames(['BTC', 'ETH', 'ADA', 'DOGE', 'DOT', 'XLM', 'UNI'])
# (3) Select Interval
dfs.select_interval(start="2021-07-1")
# (4) Create a new feature based on normalized averaged of open and close prices
dfs.create_feature(name='price', key=lambda dataframe: cp.normalize((dataframe['open'] + dataframe['close']) / 2))
# (5) Select price
dfs.select_col(['price'])
# (6) PLOT
dfs.plot(title='Returns', save='returns')
```
![image info](examples/returns.png)
##### For more examples, please refer to examples folder

# Services

- [ ] Customizable Notifier
  - How much did prices and volumes of coins change in the last X hours ?
- [ ] Trade Assistants
  -  Given a portfolio (total asset and its allocation), how would have a trade assistant suggest in the last X hours

### In Progress

- [ ] Data Acquisition

### Done ✓

- [x] Create fetching coins from Coinbase.
- [x] Provide information about coins.

# Help and Support
For any further questions or suggestions, please contact: caglardemir8@gmail.com