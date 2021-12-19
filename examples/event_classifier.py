"""
Detection of trends: Given prices of X previous hours, will price increase or decrease Y% amount within next Z hours?

(1) X = previous prices
(2) y = next prices
(3) X_norm = X \ X[0] - 1  # Normalize X
(4) y_norm = y \ X[0] - 1  # Normalize y
(5) Given X_norm, which decision should we take if we know y_norm ?
(5.1) Look at last few prices e.g. average prices of last hour X_norm[-12:].mean()
(5.2) x_decision_point = X_norm[-12:].mean()
(5.3) y_decision_point = y.mean()


(6) if y_decision_point > (x_decision_point * 1.02) and y_decision_point > 0 and x_decision_point > 0
We should buy now because prices will increase


(7) if y_decision_point =< (x_decision_point * .98) and y_decision_point > 0 and x_decision_point > 0
We should sell now because prices decreasing

(8) OTHERWISE HOLD
"""

import coinpy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from matplotlib import animation
from matplotlib.animation import FuncAnimation

# Fixing the random seeds.
torch.backends.cudnn.deterministic = True
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# Problem Definition
# INPUT: Previously seen prices
length_prev_seq = 12 * 24 * 7  # 7 DAYS: 12 x 5minutes => 1 hour
# OUTPUT: Next seen prices: Create label in the next seen data
length_next_seq = 12  # 1 Hour
# Percentage higher or lower for Multi Class Classification BUY, SELL, WAIT
percent = .02
# Configurations for learning
batch_size = 2048
num_epoch = 0

print(f'Input is {length_prev_seq / (12 * 24)} day price')
print(f'Output whether {percent * 100} % increase/decrease {length_next_seq / 12} hour')


def decision_evaluator(title: str, seq_of_decisions: list):
    budget = 0
    num_buy, num_sell = 0, 0
    print(f'Number of decision {len(seq_of_decisions)}')
    for dec, price in seq_of_decisions:
        if dec == 'BUY':
            budget -= price[0]
            num_buy += 1
        elif dec == 'SELL':
            budget += price[0]
            num_sell += 1
        else:
            raise ValueError()
    print(f'{title} on {coin_name} current budget:{budget} after Number of Buy:{num_buy}\t Number of Sell:{num_sell}')


def trade_simulator(model, coin_name, d, length_prev_seq):
    decisions = []
    plt.plot(d, alpha=.5)
    with torch.no_grad():
        for k, i in enumerate(range(len(d) - length_prev_seq)):
            prev_seq = d[i:(i + length_prev_seq)]

            x_input = torch.from_numpy(prev_seq).float().flatten()
            y_hat = model(x_input / x_input[0] - 1)
            decision = y_hat.argmax()

            # BUY, SELL or HOLD
            if decision == 0:
                # if buy_flag is True:
                decisions.append(['BUY', prev_seq[-1]])  # ALLOWED TO BUY
                # buy_flag = False
                plt.scatter(i + length_prev_seq, prev_seq[-1], c='g', marker='^')
            elif decision == 1:
                # if buy_flag is False:
                decisions.append(['SELL', prev_seq[-1]])  # ALLOWED TO SELL
                #plt.scatter(i + length_prev_seq, prev_seq[-1], c='r', marker='v')
            else:
                pass
    plt.title(f'Buy:Green, Sell:Red on {coin_name}')
    plt.show()
    print('Testing completed')
    decision_evaluator('Linear Trader', decisions)


class LinearTrader(torch.nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.af1 = nn.Linear(input_dim, output_dim)

    def forward(self, seq_input):
        logits = self.af1(seq_input)
        return torch.sigmoid(logits)

    @staticmethod
    def event_definition(seen_price, next_price):
        assert len(seen_price) > 0
        assert len(next_price) > 0

        seen_price = seen_price.flatten()
        next_price = next_price.flatten()

        # (1) Normalize seen and next prices with oldest price
        norm_seen_price = seen_price / seen_price[0] - 1
        norm_next_price = next_price / seen_price[0] - 1

        # (2) Consider only the average of last 1 hour prices for labelling
        x_decision_point = norm_seen_price[-12:].mean()

        # (5) Compute Target, mean price of next hour
        y_decision_point = norm_next_price.mean()

        label = np.zeros(3)
        # Prices go up and positive and future is brighter
        if y_decision_point >= x_decision_point * 1.02 and (y_decision_point > 0) and (x_decision_point > 0):
            label[0] = 1  # BUY
        # Prices go down and positive and future is not brighter
        elif y_decision_point <= x_decision_point * .98 and (y_decision_point > 0) and (x_decision_point > 0):
            label[1] = 1  # Sell
            """
            # Prices are negative. and negative and and future is brighter
            elif y_decision_point >= x_decision_point * .98 and (y_decision_point < 0) and (x_decision_point < 0):
                label[0] = 1  # BUY
            # Prices are negative. and negative and and future is brighter
            elif y_decision_point <= x_decision_point * 1.02 and (y_decision_point < 0) and (x_decision_point < 0):
                label[1] = 1  # Sell
            """
        else:
            label[2] = 1  # WAIT
        return seen_price, label


class NonLinearTrader(LinearTrader):
    def __init__(self, input_dim, output_dim=3):
        super().__init__(input_dim, input_dim+input_dim)
        self.model = torch.nn.Sequential(self.af1, nn.ReLU(),
                                         nn.Linear(input_dim+input_dim, input_dim),
                                         nn.ReLU(),
                                         nn.Linear(input_dim, output_dim))

    def forward(self, seq_input):
        # From Logits to Preds
        return torch.sigmoid(self.model(seq_input))

    @staticmethod
    def event_definition(seen_price, next_price):
        assert len(seen_price) > 0
        assert len(next_price) > 0

        seen_price = seen_price.flatten()
        next_price = next_price.flatten()

        # (1) Normalize seen and next prices with oldest price
        norm_seen_price = seen_price / seen_price[0] - 1
        norm_next_price = next_price / seen_price[0] - 1

        # (2) Consider only the average of last 1 hour prices for labelling
        x_decision_point = norm_seen_price[-12:].mean()

        # (5) Compute Target, mean price of next hour
        y_decision_point = norm_next_price.mean()

        label = np.zeros(3)
        # Prices go up and positive and future is brighter
        if y_decision_point >= x_decision_point * 1.02 and (y_decision_point > 0) and (x_decision_point > 0):
            label[0] = 1  # BUY
        # Prices go down and positive and future is not brighter
        elif y_decision_point <= x_decision_point * .98 and (y_decision_point > 0) and (x_decision_point > 0):
            label[1] = 1  # Sell
        # Prices are negative. and negative and and future is brighter
        elif y_decision_point >= x_decision_point * .98 and (y_decision_point < 0) and (x_decision_point < 0):
            label[0] = 1  # BUY
        # Prices are negative. and negative and and future is brighter
        elif y_decision_point <= x_decision_point * 1.02 and (y_decision_point < 0) and (x_decision_point < 0):
            label[1] = 1  # Sell
        else:
            label[2] = 1  # WAIT
        return seen_price, label


for coin_name in ['BTC', 'ADA', 'ETH', 'SOL']:
    # (1) Load dataframes holding low, high, open, close and volume of coins in 5 minutes interval
    dfs = cp.DataFramesHolder(path='../Data')
    # (2) Drop dataframes having length of less than 60 days
    dfs.drop_data_frames(key=lambda x: len(x) < 12 * 24 * 60)
    # (3) Create a feature based on the average value of open and close prices
    dfs.create_feature(name='price', key=lambda df: (df['open'] + df['close']) / 2)
    # (4) Select only price feature
    dfs.select_col(['price'])
    df = dfs[coin_name]
    n_time_stamp, n_coins = df.shape

    model = LinearTrader(input_dim=int(n_coins * length_prev_seq), output_dim=3)

    dataset = cp.EventDataset(df, seq_length=length_prev_seq, even_interval=length_next_seq,
                              labeller=model.event_definition)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    print(f'\nTRAIN a classifier on {coin_name}')
    for i in range(num_epoch):
        losses = []
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_hat = model(x)
            # Compute prediction error
            loss = loss_fn(y_hat, y)
            losses.append(loss.item())
            # Backpropagation
            loss.backward()
            optimizer.step()

        losses = np.array(losses)
        # if i % 25 == 0:
        print(f'{i}.th Epoch\t Avg. Loss:{losses.mean():.3f}\tStd. Loss:{losses.std():.3f}')

    # Important to remember index of coins.
    torch.save(model.state_dict(), f'{coin_name}_model_weights.pth')
    model.eval()
    print('Test Training on ***TRAINING DATA***')
    trade_simulator(model, coin_name, df.values[:10_000], length_prev_seq)
    break
