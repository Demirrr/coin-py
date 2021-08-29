import matplotlib.pyplot as plt
import coinpy as cp

# (1) Load dataframes holding low, high, open, close and volume of coins in 5 minutes interval
dfs = cp.DataFramesHolder(path='../Data')
# (2) Drop dataframes having length of less than 30 days
dfs.drop_data_frames(key=lambda x: len(x) < 12 * 24 * 30)
dfs.dropna()
# (3) Create a feature based on the average value of open and close prices
dfs.create_feature(name='price', key=lambda dataframe: (dataframe['open'] + dataframe['close']) / 2)
# (4) Create a feature represent real volume
dfs.create_feature(name='real_volume', key=lambda dataframe: dataframe['price'] * dataframe['volume'])
def most_profitable_coins(n, dfh, start_interval):
    print(f'\nMost Profitable Coins Since {start_interval}\n')

    # Find 10 coins whose average normalized price is the highest starting from "2021-01-1", till now
    for (c, v) in dfh.find(n=n, start=start_interval,
                           key=lambda dataframe: cp.normalize(dataframe['price']).mean(),
                           descending=True):
        print(f'{c}:\t{v:.6}')


def most_traded_coins(n, dfh, start_interval):
    print(f'\nMost Traded Coins Since {start_interval}\n')

    # Find 10 coins whose average normalized price is the highest starting from "2021-01-1", till now
    for (c, v) in dfh.find(n=n, start=start_interval,
                           key=lambda dataframe: cp.normalize(dataframe['real_volume']).mean(),
                           descending=True):
        print(f'{c}:\t{v:.6}')


def most_stable_coins(n, dfh, start_interval):
    print(f'\nMost Stable Coins Since {start_interval}\n')
    # Find 10 coins whose average normalized price are most stable  starting from "2021-01-1", till now
    for (c, v) in dfh.find(n=n, start=start_interval,
                           key=lambda dataframe: cp.normalize(dataframe['price']).std(),
                           descending=False):
        print(f'{c}:\t{v:.6}')


for start_time in ["2021-01-1", "2021-02-1", "2021-03-1", "2021-04-1",
                   "2021-05-1", "2021-06-1", "2021-07-1", "2021-08-1"]:
    most_profitable_coins(3, dfs, start_time)
    most_traded_coins(3, dfs, start_time)
    most_stable_coins(3, dfs, start_time)

"""
Most Profitable Coins Since 2021-01-1

MATIC:	1.90088
ETC:	1.8441
CLV:	1.12794

Most Traded Coins Since 2021-01-1

MASK:	18.3552
WBTC:	10.046
KNC:	9.39174

Most Stable Coins Since 2021-01-1

USDT:	0.000464492
DAI:	0.000977068
POLY:	0.0864543

Most Profitable Coins Since 2021-02-1

MATIC:	1.90088
ETC:	1.8441
CLV:	1.12794

Most Traded Coins Since 2021-02-1

MASK:	18.3552
WBTC:	10.046
KNC:	9.39174

Most Stable Coins Since 2021-02-1

USDT:	0.000464492
DAI:	0.000977068
POLY:	0.0864543

Most Profitable Coins Since 2021-03-1

ETC:	3.5551
MATIC:	1.90088
FIL:	1.48236

Most Traded Coins Since 2021-03-1

MASK:	18.3552
FIL:	4.20489
QNT:	3.58562

Most Stable Coins Since 2021-03-1

USDT:	0.000464492
DAI:	0.000483017
POLY:	0.0864543

Most Profitable Coins Since 2021-04-1

ETC:	2.84261
MATIC:	2.05272
CLV:	1.12794

Most Traded Coins Since 2021-04-1

MASK:	18.3552
MKR:	4.9805
MATIC:	4.03527

Most Stable Coins Since 2021-04-1

USDT:	0.000464492
DAI:	0.000484219
POLY:	0.0864543

Most Profitable Coins Since 2021-05-1

CLV:	1.12794
ETC:	0.675913
MATIC:	0.536349

Most Traded Coins Since 2021-05-1

MASK:	18.3552
REP:	4.58858
NMR:	4.21465

Most Stable Coins Since 2021-05-1

DAI:	0.000417273
USDT:	0.000464492
POLY:	0.0864543

Most Profitable Coins Since 2021-06-1

CLV:	1.12794
QNT:	0.421854
MASK:	0.331777

Most Traded Coins Since 2021-06-1

MASK:	18.3552
QNT:	3.58562
NU:	3.53165

Most Stable Coins Since 2021-06-1

USDT:	0.000243216
DAI:	0.00029677
POLY:	0.0864543

Most Profitable Coins Since 2021-07-1

CLV:	1.12794
QNT:	0.783397
SNX:	0.411656

Most Traded Coins Since 2021-07-1

MLN:	1170.42
CRV:	61.4495
SNX:	54.2899

Most Stable Coins Since 2021-07-1

USDT:	0.000245465
DAI:	0.000279785
POLY:	0.0864543

Most Profitable Coins Since 2021-08-1

CTSI:	0.532914
ADA:	0.51773
SOL:	0.504478

Most Traded Coins Since 2021-08-1

CTSI:	6.31524
FORTH:	4.83521
OXT:	3.84593

Most Stable Coins Since 2021-08-1

USDT:	0.000261817
DAI:	0.000297845
REP:	0.0333664
"""