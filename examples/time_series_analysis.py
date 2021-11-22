import coinpy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/BTC.csv', parse_dates=['time'])
# (1) Add price
df['price'] = (df['open'] + df['close']) / 2
def price_visualize(data):
    # Visualize Time series
    data=data.set_index('time')
    # (1) Visualize the BTC prices
    data['price'].plot()
    plt.title('BTC Prices')
    plt.show()
    for t in ['H','D','W','M']:
        # (2) Visualize Average BTC price per month
        data['price'].resample(t).mean().plot(label='Mean')
        # (2) Visualize Average BTC price per month
        data['price'].resample(t).min().plot(label='Min')
        # (2) Visualize Average BTC price per month
        data['price'].resample(t).max().plot(label='Max')
        plt.title(f'BTC prices per {t}')
        plt.legend()
        plt.show()

def description(data):
    # (2) Earliest and latest date
    print(f'Earliest date : {data["time"].min()}\t Latest date: {data["time"].max()}')
    # (3) Number of days
    print(f'Number of days:{df["time"].max() - df["time"].min()}')
    # (4) Extract and add month info from time info# available infos year, weekofyear, quarter from Timestamp
    data["month"] = data["time"].dt.month

    # (5) What is the average price for each month for each quarter?
    # (5.1) Group months into quarters
    # (5.2) Report average prices
    print(data.groupby([data["time"].dt.quarter, "month"])["price"].mean())

    # (5) What is the average price for each month for each month (Not very smart but illustrate the usage of groupby method)?
    # (5.1) Group months into quarters
    # (5.2) Report average prices
    print(data.groupby([data["time"].dt.month, "month"])["price"].mean())

    # (5) What is the average price for each day of the week for each month ?
    print(data.groupby([df["time"].dt.month, "month"])["price"].mean())

    # (6) Plot Average prices
    fig, axs = plt.subplots(figsize=(12, 4))
    data.groupby(data["time"].dt.month)["price"].mean().plot(kind='bar', rot=0, ax=axs)
    plt.xlabel("Months")
    plt.ylabel("Prices $ \$$")
    plt.title('Average BTC price per month')
    plt.show()


    # (7) Plot Average prices
    fig, axs = plt.subplots(figsize=(12, 4))
    data.groupby(data["time"].dt.isocalendar().week)["price"].mean().plot(kind='bar', rot=0, ax=axs)
    plt.xlabel("Weeks")
    plt.ylabel("Prices $ \$$")
    plt.title('Average BTC price')
    plt.show()


price_visualize(df)
description(df)