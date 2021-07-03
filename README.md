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
# (2) Create a new feature based on average of open and close
dfs.preprocess({'func': 'mean', 'input': ['open', 'close'], 'output': 'price'})
# (3) Select price
dfs.select_col(['price'])
# (4) Select coins
dfs.select_frames(['BTC', 'ADA', 'ETH', 'XLM', 'UNI'])
# (5) Select Interval
dfs.select_interval(start="2021-03-25")
# (6) Normalize prices
dfs.normalize()
# (7) PLOT
dfs.plot(title='Daily Returns', save='daily_returns')
```
![image info](examples/daily_returns.png)
##### For more examples, please refer to examples folder

### Todos

- [ ] Data Modeling
- [ ] Reinforcement Learning Agent as Trader

### In Progress

- [ ] Data Acquisition

### Done ✓

- [x] Create fetching coins from Coinbase.
- [x] Provide previous information about coins.

# Help and Support
For any further questions or suggestions, please contact: caglardemir8@gmail.com