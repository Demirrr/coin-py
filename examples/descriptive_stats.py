import matplotlib.pyplot as plt
import coinpy as cp

# (1) Load dataframes holding low, high, open, close and volume of coins in 5 minutes interval
dfs = cp.DataFramesHolder(path='../Data')
# (2) Drop dataframes having length of less than 60 days
dfs.drop_data_frames(key=lambda x: len(x) < 12 * 24 * 60)
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
    dfs.beta(x='BTC', y='ETH', plot=True)
    dfs.beta(x='BTC', y='ADA', plot=True)
    dfs.beta(x='BTC', y='DOGE', plot=True)
    dfs.beta(x='XLM', y='ETH', plot=True)
    dfs.beta(x='BTC', y='UNI', plot=True)

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

"""
# OUTPUT
1.07 BTC + 0.00=ETH
0.97 BTC + 0.01=ADA
1.12 BTC -0.01=DOGE
0.62 XLM + 0.00=ETH
1.22 BTC + 0.00=UNI
96.01 USDT + 0.01=MATIC
81.28 USDT -0.00=MKR
73.74 USDT + 0.00=GTC
72.99 USDT -0.01=ICP
59.94 USDT + 0.00=SUSHI
52.06 USDT -0.00=TRB
50.03 USDT -0.00=FIL
48.81 USDT + 0.00=CHZ
48.35 USDT + 0.00=CGLD
46.39 USDT + 0.01=RLC
XLM : UNI: 0.860
XLM : ETH: 0.441
XLM : BTC: 0.726
ETH : UNI: 0.488
BTC : UNI: 0.820
BTC : ETH: 0.080


Top 10 positive correlation
1. BTC : WBTC: 1.000
2. ZRX : BAT: 0.992
3. GRT : REN: 0.986
4. REN : BAT: 0.986
5. BNT : BAT: 0.985
6. LTC : TRB: 0.984
7. BNT : TRB: 0.984
8. BAND : BAT: 0.984
9. BCH : TRB: 0.981
10. NKN : CHZ: 0.980


Top 10 negative correlation
1. MATIC : BAND: -0.543
2. ETC : NU: -0.547
3. MATIC : SKL: -0.556
4. BTC : MATIC: -0.594
5. MATIC : STORJ: -0.597
6. MATIC : OXT: -0.602
7. MATIC : REN: -0.607
8. MATIC : NU: -0.611
9. MATIC : WBTC: -0.638
10. MATIC : GRT: -0.642
"""