from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prophet import Prophet
# https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/blob/master/StockPricesPredictionProject/pricePredictionLSTM.py
# https://github.com/stefan-jansen/machine-learning-for-trading
# https://github.com/mfrdixon/ML_Finance_Codes
# https://github.com/matplotlib/mplfinance
# https://farid.one/kaggle-solutions/

def read_csv(path,name):
    df=pd.read_csv(path,index_col='time', parse_dates=True).add_suffix(f'_{name}')
    df.drop_duplicates(inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    df[f'price_{name}'] = (df[f'open_{name}'] + df[f'close_{name}']) / 2
    return df[[f'price_{name}',f'volume_{name}']]
def read_merge_data_frames(paths_names):
    df=None
    for path,name in paths_names:
        if df is None:
            df=read_csv(path,name)
        else:
            df = df.join(read_csv(path,name), how='outer')
    return df
# (1) READ and Merge data
df=read_merge_data_frames([('../Data/BTC.csv','btc'),
                           ('../Data/ETH.csv','eth')])

for name in ['BTC','ETH','MATIC']:
    df=pd.read_csv(f'../Data/{name}.csv',index_col='time', parse_dates=True)
    df.drop_duplicates(inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    # Create Price value
    df['y'] = (df[f'open'] + df[f'close']) / 2

    # Resample data by by the mean of the day
    df_daily = df.resample('D').mean()
    # (2) Prepare Data : ds:y
    df_daily['ds']=df_daily.index
    df_daily_prophet=df_daily[['ds','y']]
    df_daily_prophet=df_daily_prophet.reset_index()
    df_daily_prophet=df_daily_prophet.drop(columns=['time'])

    m = Prophet()
    m.fit(df_daily_prophet)
    future = m.make_future_dataframe(periods=10)

    # Python
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(50))

    fig1 = m.plot(forecast)
    plt.title(name)
    plt.show()

