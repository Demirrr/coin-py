from argparse import ArgumentParser
from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly,plot_yearly
from prophet.plot import add_changepoints_to_plot

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
            df_daily = df
        # (2) Prepare Data : ds:y
        df_daily['ds'] = df_daily.index
        df_daily_prophet = df_daily[['ds', 'y']]
        df_daily_prophet = df_daily_prophet.reset_index()
        df_daily_prophet = df_daily_prophet.drop(columns=['time'])

        m = Prophet(changepoint_prior_scale=args.changepoint_prior_scale,
                    interval_width=args.interval_width,
        mcmc_samples=args.mcmc_samples
                    )
        m.add_country_holidays(country_name='US')
        m.fit(df_daily_prophet)
        # TODO: Adding other time series => https://nbviewer.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb
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
            predictions.to_csv(f'Pred_' + name + '.csv')

        fig = m.plot(forecast,  # figsize=(14, 8),
                     ylabel=f'{name}-price')
        plt.title(f'{name}-price')

        if args.add_changepoints_to_plot:
            a = add_changepoints_to_plot(fig.gca(), m, forecast)
        if args.save_plot:
            plt.savefig(name)
        plt.show()
        if args.plot_component:
            fig = m.plot_components(forecast)
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--averaging_interval", type=str,
                        default='6H',
                        help='[1H,6H,D,W,M,None]')
    parser.add_argument("--plot_component", default=False)
    parser.add_argument("--save_plot", default=True)
    parser.add_argument("--save_predictions", default=True)
    parser.add_argument("--add_changepoints_to_plot", default=True)
    parser.add_argument("--num_preds", type=int, default=30)
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05,
                        help='If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument')
    parser.add_argument("--interval_width", type=float, default=0.80,
                        help='')
    parser.add_argument("--mcmc_samples", type=int, default=0,
                        help='mcmc_samples: Integer, if greater than 0, will do full Bayesian inference with the specified number of MCMC samples. If 0, will do MAP estimation.')


    run(parser.parse_args())
