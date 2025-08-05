import datetime
import pandas as pd
import numpy as np


def load_requests():
    df = pd.read_csv('C:\\Users\\saoudi\\Desktop\\final\\Data\\10min_all_data_new.csv', parse_dates=['time'], index_col=['time'])
    return df



def load_day(df, day, max_steps):
    if day is None:
        day = 150  # Set a default value or handle it in a way that makes sense for your use case

    minutes = max_steps * 10
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.datetime(2022, 1, 1) + pd.DateOffset(days=day - 1)
    end_date = start_date + time_delta
    df = df.loc[(df.index >= start_date) & (df.index < end_date)]
    return df


def get_device_demands(df, agent_ids, day, timestep):
    if day is None:
        day = 150  # Set a default value or handle it in a way that makes sense for your use case

    minutes = timestep * 10
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.datetime(2022, 1, 1) + pd.DateOffset(days=day - 1)
    time = start_date + time_delta
    df = df[(df['dataid'].isin(agent_ids)) & (df.index == time)]
    return df


def get_peak_demand(df):
    df = df.groupby(pd.Grouper(freq='10Min')).sum()
    return df['total'].max()


def load_baselines():
    return np.load('C:\\Users\\saoudi\\Desktop\\final\\Data\\baselines.npy')