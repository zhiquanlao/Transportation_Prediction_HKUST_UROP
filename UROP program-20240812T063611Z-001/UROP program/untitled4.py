# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:23:20 2021

@author: CHEN
"""
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,plot_importance
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt
import datetime
weekdays=["MON","TUE","WED","THU","FRI"]
def calculateTime(time):
    return (time-np.datetime64(time,'D'))/np.timedelta64(1,'h')
def change(row):
    return calculateTime(row['date_time'])
#return a dataframe containing X and y, with date_time as index
def retrieve_data(day):
    y=pd.DataFrame()
    all_days=pd.date_range("20170101","20171231",freq="W-"+weekdays[day])
    all_days=all_days.strftime('%Y%m%d')
    y_date=pd.DataFrame()
    for date in all_days:
        path="C:/Users/CHEN/Desktop/UROP_data/Output_daily_zip/"+date+"_output_daily_df.pickle"
        daily_df=pd.read_pickle(path)
        daily_df=daily_df[daily_df.index.hour!=23]
        daily_df=daily_df.reset_index()
        y_date=pd.concat([y_date,daily_df['index']],ignore_index=True)
        daily_df=daily_df['339_509_median_latency_5']
        y=pd.concat([y,daily_df],ignore_index=True)
    y=y.rename(columns={0:"339_509_median_latency_5"})
    y=y[~y.index.duplicated()]
    y_date=y_date.rename(columns={0:"date_time"})
    y_date=y_date[~ y_date.index.duplicated()]
    #retrive median_speed
    all_in=pd.read_pickle("C:/Users/CHEN/Desktop/UROP_data/all_in_out/all_in_out.pickle")
    all_in=all_in[all_in.index.weekday==day]
    all_in=all_in[all_in.index.hour!=23]
    all_in=all_in.reset_index(drop=True)
    col_to_get=['339_376_median_speed_past_5','339_376_median_speed_past_10','339_376_median_speed_past_15','339_413_median_speed_past_5','339_413_median_speed_past_10','339_413_median_speed_past_15','339_467_median_speed_past_10','339_467_median_speed_past_15','339_509_median_speed_past_10','339_509_median_speed_past_15','376_509_median_speed_past_10','376_509_median_speed_past_15']
    all_in=all_in[col_to_get]
    all_in=all_in[~all_in.index.duplicated()]
    #retrieve flux
    col_to_get=['339_out_p1','339_out_p2','339_out_p3','413_out_p1','413_out_p2','413_out_p3','acc_tot','413_acc']
    df=pd.read_pickle("C:/Users/CHEN/Desktop/UROP_data/df_new_ver5_20190401.pickle")
    df=df[df.index.day_of_week==day]
    df=df[df.index.hour!=23]
    df=df.reset_index(drop=True)
    df=df[df.index<len(y)]
    df=df[col_to_get]
    df=df[~df.index.duplicated()]
    #retrieve predicted latency
    pl=pd.read_pickle("C:/Users/CHEN/Desktop/UROP_data/pl_inputs.pickle")
    pl=pl[pl.index.weekday==day]
    pl=pl[pl.index.hour!=23]
    pl=pl.reset_index(drop=True)
    pl=pl['339_509_p_l_median']
    pl=pl[~pl.index.duplicated()]
    #concat
    X=pd.concat([df,pl,all_in,y_date,y],axis=1)
    X['time']=X.apply(lambda row:change(row),axis=1)
    X=X.set_index('date_time')
    X=X.dropna()
    return X
X=retrieve_data(0)
number =1
"""
#=============separate date=======================
all_days=np.unique(X.index.date)
np.random.shuffle(all_days)
test_date,temp_date=all_days[:8],all_days[8:]
np.random.shuffle(temp_date)
ver_date,train_date=temp_date[:8],temp_date[8:]
X['temp']=X.index.date
X_train=X[X['temp'].isin(train_date)]
X_ver=X[X['temp'].isin(ver_date)]
X_test=X[X['temp'].isin(test_date)]
X=X.drop('temp',axis=1)
X_train=X_train.drop('temp',axis=1)
X_test=X_test.drop('temp',axis=1)
X_ver=X_ver.drop('temp',axis=1)
#==============================================
"""