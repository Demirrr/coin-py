import pandas as pd

import coinpy as cp
from coinpy.util import get_all_folders, get_all_data, load_csv
import os
import datetime
import shutil


def create_folder(directory):
    if os.path.exists(directory):
        """ File found. Do nothing """
    else:
        """ File not found. Create one  nothing """
        print(f"Create folder {directory}")
        os.mkdir(directory)


def fetch_files(path, suffix):
    # 1. Get all first level folders under path.
    for i in get_all_folders(path):
        # 2. For each fetch data folder, collect csv files.
        for j in get_all_folders(i):
            if suffix in j:
                coin_csv_file = j[j.rfind('/') + 1:-4]
                yield coin_csv_file


def fetch_data():
    # (1) Fetch the data save it into Fetched Folder with current time
    cp.Collector().fetch_and_save(granularity=60, path='/home/demir/Desktop/Demir')
    path_fetched = '/home/demir/Desktop/Demir/Fetched'
    # (2) Merge (1) with previous data
    path_hourly_update = '/home/demir/Desktop/Demir/HourlyUpdate'
    create_folder(path_hourly_update)
    suffix = '-USD.csv'
    for i in fetch_files(path_fetched, suffix):
        df = get_all_data(data=i, path=path_fetched)
        # Remove suffix at saving
        key = suffix.replace('.csv', '')
        coin_name = i.replace(key, '')
        path_of_folder_coin_name = f'{path_hourly_update}/{coin_name}'
        create_folder(directory=path_of_folder_coin_name)
        if os.path.exists(f'{path_of_folder_coin_name}/' + coin_name + '.csv'):
            """ File found. Do nothing """
            df_main = pd.read_csv(f'{path_of_folder_coin_name}/' + coin_name + '.csv', index_col=0)
            print(df_main.shape)
            df_main = df_main.append(df, verify_integrity=False)
            df_main.drop_duplicates(inplace=True)
            print(df_main.shape)
            df_main.to_csv(f'{path_of_folder_coin_name}/' + coin_name + '.csv')
        else:
            """ File not found. Create one  nothing """
            df.to_csv(f'{path_of_folder_coin_name}/' + coin_name + '.csv')

    # (3) Delete Fetched Data
    shutil.rmtree(path_fetched)


fetch_data()
exit(1)

from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()


@sched.scheduled_job('interval', minutes=1)
def coin_news():
    # (1) Get new data
    # (2) Check existing data
    # (3) Compare how prices are changed in the last hour
    # (4) Write an email to the person
    print('This job is run every hour seconds.')


@sched.scheduled_job('interval', seconds=3)
def timed_job():
    # Fetch Data
    print('This job is run every 10 seconds.')


@sched.scheduled_job('cron', day_of_week='mon-fri', hour=19)
def scheduled_job():
    print('This job is run every weekday at 10am.')


sched.configure()
sched.start()

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


for start_time in ["2021-08-1"]:
    most_profitable_coins(10, dfs, start_time)
    # most_traded_coins(3, dfs, start_time)
    # most_stable_coins(3, dfs, start_time)
