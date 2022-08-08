from argparse import ArgumentParser
from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly, plot_yearly
from prophet.plot import add_changepoints_to_plot

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE
import os
import glob


# Study
# => https://pytorch-forecasting.readthedocs.io/en/stable/index.html
# => https://github.com/nicolasfauchereau/Auckland_Cycling/blob/master/code/utils.py
# and https://nbviewer.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb
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


def prepare_data_frame(path: str):
    df = pd.read_csv(path, index_col='time', parse_dates=True)
    df.drop_duplicates(inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    # Create Price value
    df['y'] = (df[f'open'] + df[f'close']) / 2
    return df


def preprocess_data_frame(df: pd.DataFrame, args) -> pd.DataFrame:
    """ Modify input data for fitting """
    # Resample data by the mean of the day
    if args.averaging_interval:
        df_daily = df.resample(args.averaging_interval).mean()
    else:
        df_daily = df
    # (2) Prepare Data : ds:y
    df_daily['ds'] = df_daily.index
    df_daily_prophet = df_daily[['ds', 'y']]
    df_daily_prophet = df_daily_prophet.reset_index()
    df_daily_prophet = df_daily_prophet.drop(columns=['time'])
    return df_daily_prophet


def fit(df, args):
    m = Prophet(changepoint_prior_scale=args.changepoint_prior_scale,
                interval_width=args.interval_width,
                mcmc_samples=args.mcmc_samples
                )
    m.add_country_holidays(country_name='US')
    # m.add_regressor('low', prior_scale=0.5, mode='multiplicative') ?
    print(f'Model is trained on {len(df)} data')
    m.fit(df)
    return m


def make_verif(forecast, data_train, data_test):
    """
    Put together the forecast (coming from fbprophet)
    and the overved data, and set the index to be a proper datetime index,
    for plotting
    Parameters
    ----------
    forecast : pandas.DataFrame
        The pandas.DataFrame coming from the `forecast` method of a fbprophet
        model.

    data_train : pandas.DataFrame
        The training set, pandas.DataFrame
    data_test : pandas.DataFrame
        The training set, pandas.DataFrame

    Returns
    -------
    forecast :
        The forecast DataFrane including the original observed data.
    """

    forecast.index = pd.to_datetime(forecast.ds)

    data_train.index = pd.to_datetime(data_train.ds)

    data_test.index = pd.to_datetime(data_test.ds)

    data = pd.concat([data_train, data_test], axis=0)

    forecast.loc[:, 'y'] = data.loc[:, 'y']

    return forecast


def plot_verif(verif, iloc):
    """
    plots the forecasts and observed data, the `year` argument is used to visualise
    the division between the training and test sets.
    Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package
    year : integer
        The year used to separate the training and test set. Default 2017
    Returns
    -------
    f : matplotlib Figure object
    """

    f, ax = plt.subplots(figsize=(14, 8))
    train = verif.iloc[:-iloc, :]

    ax.plot(train.index, train.y, 'ko', markersize=3)

    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)

    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)

    test = verif.iloc[-iloc:, :]

    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)

    ax.axvline(test.index[0], color='0.8', alpha=0.7)

    ax.grid(ls=':', lw=0.5)

    return f


def data_split(df, args):
    data_train = df.iloc[:-args.num_preds, :]
    data_test = df.iloc[-args.num_preds:, :]

    data_train.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)
    return data_train, data_test


def plot_joint_plot(verif, x='yhat', y='y', title=None, fpath='../figures/paper', fname=None):
    """

    Parameters
    ----------
    verif : pandas.DataFrame
    x : string
        The variable on the x-axis
        Defaults to `yhat`, i.e. the forecast or estimated values.
    y : string
        The variable on the y-axis
        Defaults to `y`, i.e. the observed values
    title : string
        The title of the figure, default `None`.

    fpath : string
        The path to save the figures, default to `../figures/paper`
    fname : string
        The filename for the figure to be saved
        ommits the extension, the figure is saved in png, jpeg and pdf

    Returns
    -------
    f : matplotlib Figure object
    """
    g = sns.jointplot(x='yhat', y='y', data=verif, kind="reg", color="0.4")

    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]

    if title is not None:
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]

    # ax.set_xlim([-5, None])
    # ax.set_ylim([-5, 3000])

    corr = verif.loc[:, ['y', 'yhat']].corr()
    print(corr)
    r = corr.iloc[0, 1]
    mean_absolute_error = np.nansum(np.absolute(verif.loc[:, 'y'].values - verif.loc[:, 'yhat'].values)) / len(verif)
    ax.text(5, 5, f"R = {r:+4.2f}\nMAE = {mean_absolute_error:4.1f}", fontsize=15)

    ax.set_xlabel("model's estimates", fontsize=15)

    ax.set_ylabel("observations", fontsize=15)

    ax.grid(ls=':')

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.grid(ls=':')

    # if fname is not None:
    #    for ext in ['png', 'jpeg', 'pdf']:
    #        g.fig.savefig(os.path.join(fpath, "{}.{}".format(fname, ext)), dpi=200)


def predict(path, args):
    name = path.split('/')[1][:-4]
    df = prepare_data_frame(path)
    try:
        df = preprocess_data_frame(df, args)
    except TypeError as e:
        print(e)
        return None
    data_train, data_test = data_split(df, args)
    # Fit the model
    try:
        m = fit(data_train, args)
    except ValueError as e:
        print(e)
        return None
    # Make a prediction
    future = m.make_future_dataframe(periods=args.num_preds, freq=args.averaging_interval)
    forecast = m.predict(future)
    verif = make_verif(forecast, data_train, data_test)
    plot_verif(verif, args.num_preds)
    plt.title(f'{name} Prediction on avg of {args.averaging_interval} interval')
    plt.savefig('figures/' + name)
    # plt.show()
    if args.plot_joint:
        plot_joint_plot(verif.iloc[:-args.num_preds, :], title='train set')
        plt.show()

        plot_joint_plot(verif.iloc[-args.num_preds:, :], title='test set')
        plt.show()
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(args.num_preds)
    if args.save_predictions:
        predictions.to_csv(f'Pred_' + name + '.csv')

    # fig = m.plot(forecast,ylabel=f'{name}-price')
    # plt.title(f'{name}-price')

    if args.add_changepoints_to_plot:
        add_changepoints_to_plot(fig.gca(), m, forecast)
    if args.save_plot:
        plt.savefig(name)
    # plt.show()
    if args.plot_component:
        m.plot_components(forecast)
        plt.show()


def run(args):
    #  https://nbviewer.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb#incorporating-the-effects-of-weather-conditions
    if len(args.main_dataset_path) > 0:
        for p in args.main_dataset_path:
            predict(p, args)
    else:
        for p in glob.glob("Data/*.csv"):
            predict(p, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--main_dataset_path",  # name on the CLI - drop the `--` for positional/required parameters
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=['Data/ETH.csv', 'Data/BTC.csv', 'Data/SOL.csv', 'Data/AVAX.csv',
                                 'Data/MATIC.csv', 'Data/ATOM.csv', 'Data/MANA.csv', 'Data/AXS.csv',
                                 'Data/GALA.csv', 'Data/LRC.csv'],  # default if nothing is provided
                        # default=[],  # If empty, work on all data
                        )
    parser.add_argument("--averaging_interval", type=str,
                        default='6H',
                        help='[1H,6H,D,W,M,None]')

    parser.add_argument("--plot_joint", default=True)
    parser.add_argument("--plot_component", default=False)
    parser.add_argument("--save_plot", default=False)
    parser.add_argument("--save_predictions", default=False)
    parser.add_argument("--add_changepoints_to_plot", default=False)
    parser.add_argument("--num_preds", type=int, default=int(7 * 4))
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05,
                        help='If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument')
    parser.add_argument("--interval_width", type=float, default=0.95,
                        help='')
    parser.add_argument("--mcmc_samples", type=int, default=0,
                        help='mcmc_samples: Integer, if greater than 0, will do full Bayesian inference with the specified number of MCMC samples. If 0, will do MAP estimation.')

    run(parser.parse_args())
