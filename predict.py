from argparse import ArgumentParser
from prophet import Prophet
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/blob/master/StockPricesPredictionProject/pricePredictionLSTM.py
# https://github.com/stefan-jansen/machine-learning-for-trading
# https://github.com/mfrdixon/ML_Finance_Codes
# https://github.com/matplotlib/mplfinance
# https://farid.one/kaggle-solutions/


def read_csv(path, name):
    df = pd.read_csv(path, index_col='time', parse_dates=True).add_suffix(f'_{name}')
    df.drop_duplicates(inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    df[f'price_{name}'] = (df[f'open_{name}'] + df[f'close_{name}']) / 2
    return df[[f'price_{name}', f'volume_{name}']]


def read_merge_data_frames(paths_names):
    df = None
    for path, name in paths_names:
        if df is None:
            df = read_csv(path, name)
        else:
            df = df.join(read_csv(path, name), how='outer')
    return df


def run(args):
    for name in ['BTC', 'ETH', 'MATIC']:
        df = pd.read_csv(f'Data/{name}.csv', index_col='time', parse_dates=True)
        df.drop_duplicates(inplace=True)
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
        # Create Price value
        df['y'] = (df[f'open'] + df[f'close']) / 2

        # Resample data by by the mean of the day
        if args.averaging_interval:
            df_daily = df.resample(args.averaging_interval).mean()
        else:
            df_daily=df
        # (2) Prepare Data : ds:y
        df_daily['ds'] = df_daily.index
        df_daily_prophet = df_daily[['ds', 'y']]
        df_daily_prophet = df_daily_prophet.reset_index()
        df_daily_prophet = df_daily_prophet.drop(columns=['time'])

        m = Prophet()
        m.fit(df_daily_prophet)
        print(f'Model is trained on {len(df_daily_prophet)} data')
        future = m.make_future_dataframe(periods=args.num_preds)
        # Python
        forecast = m.predict(future)
        # forecast.columns: ['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
        #        'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
        #        'weekly', 'weekly_lower', 'weekly_upper', 'multiplicative_terms',
        #        'multiplicative_terms_lower', 'multiplicative_terms_upper', 'yhat']
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(args.num_preds)
        if args.save_predictions:
            predictions.to_csv(f'Pred_'+name + '.csv')
        m.plot(forecast, figsize=(14, 8), ylabel=f'{name}-price')
        if args.save_plot:
            plt.savefig(name)
        plt.show()
        if args.plot_component:
            fig2 = m.plot_components(forecast)
            plt.title(name + 'Component')
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--averaging_interval", type=str,
                        default=None,#'D',
                        help='[D,W,None]')
    parser.add_argument("--plot_component", default=None)
    parser.add_argument("--save_plot", default=True)
    parser.add_argument("--save_predictions", default=True)
    parser.add_argument("--num_preds", type=int,default=50)
    run(parser.parse_args())
