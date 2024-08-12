# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:46:33 2022

@author: CHEN
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import datetime
import time
import csv
import matplotlib.pyplot as plt
from random import shuffle

def sep_weekday(input_df, weekday=[]):
    global df_wd
    df_wd_all = pd.DataFrame()
    
    for day in weekday:
        df_wd = input_df.loc[input_df.index.weekday == day]
        df_wd_all = pd.concat([df_wd_all, df_wd])
    return df_wd_all



def sep_train_test(input_df, weekday, testing_days):
#     dropping the testing days from the input_df
    df_train = input_df
    
    testing_days_lt = []
    for i in weekday:
        testing_days_lt += testing_days[i]
        
    for i in testing_days_lt:
        df_train = df_train[df_train.index.date != i]
        
#     Combining the testing days into a dataframe
    df_test = pd.DataFrame()
    for i in testing_days_lt:
        df_test = pd.concat([df_test, input_df[input_df.index.date == i]])
        
    return df_train, df_test

def XGB():
    global df_train_pred
    global df_test_pred
    X_train_new = df_train[input_columns_new]
    X_test_new = df_test[input_columns_new]

    y_train_new = df_train[target_columns]
    y_test_new = df_test[target_columns]
    """
    X_train_new = X_train_new[input_columns_new]
    X_test_new  = X_test_new[input_columns_new]

    y_train_new = y_train_new[target_columns]
    y_test_new = y_test_new[target_columns]
    """

    pred_model = xgb.XGBRegressor()
    pred_model.fit(X_train_new, y_train_new)
    y_train_pred = pred_model.predict(X_train_new)
    X_train_new['pred_new'] = y_train_pred
    y_test_pred = pred_model.predict(X_test_new)
    X_test_new['pred_new'] = y_test_pred
    
    df_train_pred['pred_new']  =   X_train_new['pred_new']
    df_test_pred['pred_new']   =   X_test_new['pred_new']
def GMM(no_of_cluster):
    for i in range(no_of_cluster):
        global df_train_pred
        global df_test_pred
        X_train_new = df_train[df_train['GMM'] == i]    #for cluster 0
        X_test_new = df_test[df_test['GMM'] == i]

        y_train_new = df_train[df_train['GMM'] == i]
        y_test_new = df_test[df_test['GMM'] == i]

        X_train_new = X_train_new[input_columns_new]
        X_test_new  = X_test_new[input_columns_new]

        y_train_new = y_train_new[target_columns]
        y_test_new = y_test_new[target_columns]


        pred_model = xgb.XGBRegressor()
        pred_model.fit(X_train_new, y_train_new)
        y_train_pred = pred_model.predict(X_train_new)
        X_train_new['pred_new'] = y_train_pred
        y_test_pred = pred_model.predict(X_test_new)
        X_test_new['pred_new'] = y_test_pred
        
        if i==0:
            df_train_pred  =   X_train_new
            df_test_pred   =   X_test_new
        else:
            df_train_pred = pd.concat([df_train_pred, X_train_new])
            df_test_pred  = pd.concat([df_test_pred, X_test_new])
def predict_delay():
    global df_train_pred
    global df_test_pred
    
    X_train_new=df_train[delay_input]
    X_test_new=df_test[delay_input]
    y_train_new = df_train[tar]
    y_test_new = df_test[tar]



    pred_model = xgb.XGBRegressor()
    pred_model.fit(X_train_new, y_train_new)
    y_train_pred = pred_model.predict(X_train_new)
    X_train_new['delay_pred_new'] = y_train_pred
    y_test_pred = pred_model.predict(X_test_new)
    X_test_new['delay_pred_new'] = y_test_pred
    
    df_train_pred['delay_pred_new']  =   X_train_new['delay_pred_new']
    df_test_pred['delay_pred_new']   =   X_test_new['delay_pred_new']
##################################################################################
"""
df = pd.read_pickle("C:/Users/homan/Documents/UROP/latency/inputs_155-509/out_acc_inputs.pickle")    # import the dataframe
start_time=time.time()
# Each row represents the time. Columns are the input features and the target. 
df.head()
"""
month = 12
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
"""
# Select 2017 data
df = df.loc[df.index.year == 2017]
df = df.loc[df.index.month <= month]
"""
testing_days1 = [[datetime.date(2017, 1, 16), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 3), datetime.date(2017, 5, 15), datetime.date(2017, 6, 19), datetime.date(2017, 7, 10), datetime.date(2017, 8, 7), datetime.date(2017, 9, 18), datetime.date(2017, 10, 16), datetime.date(2017, 11, 13), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 28), datetime.date(2017, 3, 28), datetime.date(2017, 4, 18), datetime.date(2017, 5, 16), datetime.date(2017, 6, 20), datetime.date(2017, 7, 25), datetime.date(2017, 8, 22), datetime.date(2017, 9, 19), datetime.date(2017, 10, 10), datetime.date(2017, 11, 14), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 15), datetime.date(2017, 3, 1), datetime.date(2017, 4, 19), datetime.date(2017, 5, 17), datetime.date(2017, 6, 14), datetime.date(2017, 7, 12), datetime.date(2017, 8, 16), datetime.date(2017, 9, 13), datetime.date(2017, 10, 18), datetime.date(2017, 11, 1), datetime.date(2017, 12, 20)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 16), datetime.date(2017, 3, 30), datetime.date(2017, 4, 27), datetime.date(2017, 5, 11), datetime.date(2017, 6, 1), datetime.date(2017, 7, 13), datetime.date(2017, 8, 24), datetime.date(2017, 9, 14), datetime.date(2017, 10, 19), datetime.date(2017, 11, 2), datetime.date(2017, 12, 14)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 3), datetime.date(2017, 3, 17), datetime.date(2017, 4, 14), datetime.date(2017, 5, 5), datetime.date(2017, 6, 23), datetime.date(2017, 7, 14), datetime.date(2017, 8, 4), datetime.date(2017, 9, 15), datetime.date(2017, 10, 20), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 7), datetime.date(2017, 2, 4), datetime.date(2017, 3, 4), datetime.date(2017, 4, 22), datetime.date(2017, 5, 6), datetime.date(2017, 6, 3), datetime.date(2017, 7, 8), datetime.date(2017, 8, 19), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 18), datetime.date(2017, 12, 2)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 12), datetime.date(2017, 3, 12), datetime.date(2017, 4, 23), datetime.date(2017, 5, 14), datetime.date(2017, 6, 11), datetime.date(2017, 7, 16), datetime.date(2017, 8, 27), datetime.date(2017, 9, 3), datetime.date(2017, 10, 15), datetime.date(2017, 11, 19), datetime.date(2017, 12, 10)]]
testing_days2 = [[datetime.date(2017, 1, 2), datetime.date(2017, 2, 6), datetime.date(2017, 3, 27), datetime.date(2017, 4, 10), datetime.date(2017, 5, 1), datetime.date(2017, 6, 5), datetime.date(2017, 7, 24), datetime.date(2017, 8, 28), datetime.date(2017, 9, 4), datetime.date(2017, 10, 23), datetime.date(2017, 11, 27), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 7), datetime.date(2017, 3, 7), datetime.date(2017, 4, 25), datetime.date(2017, 5, 9), datetime.date(2017, 6, 13), datetime.date(2017, 7, 18), datetime.date(2017, 8, 1), datetime.date(2017, 9, 12), datetime.date(2017, 10, 24), datetime.date(2017, 11, 28), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 22), datetime.date(2017, 3, 15), datetime.date(2017, 4, 5), datetime.date(2017, 5, 3), datetime.date(2017, 6, 28), datetime.date(2017, 7, 5), datetime.date(2017, 8, 9), datetime.date(2017, 9, 20), datetime.date(2017, 10, 4), datetime.date(2017, 11, 22), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 23), datetime.date(2017, 3, 16), datetime.date(2017, 4, 13), datetime.date(2017, 5, 18), datetime.date(2017, 6, 29), datetime.date(2017, 7, 20), datetime.date(2017, 8, 10), datetime.date(2017, 9, 7), datetime.date(2017, 10, 5), datetime.date(2017, 11, 9), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 27), datetime.date(2017, 2, 10), datetime.date(2017, 3, 10), datetime.date(2017, 4, 28), datetime.date(2017, 5, 12), datetime.date(2017, 6, 16), datetime.date(2017, 7, 7), datetime.date(2017, 8, 25), datetime.date(2017, 9, 22), datetime.date(2017, 10, 27), datetime.date(2017, 11, 17), datetime.date(2017, 12, 22)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 11), datetime.date(2017, 3, 25), datetime.date(2017, 4, 1), datetime.date(2017, 5, 13), datetime.date(2017, 6, 10), datetime.date(2017, 7, 22), datetime.date(2017, 8, 12), datetime.date(2017, 9, 2), datetime.date(2017, 10, 21), datetime.date(2017, 11, 11), datetime.date(2017, 12, 16)], [datetime.date(2017, 1, 8), datetime.date(2017, 2, 19), datetime.date(2017, 3, 26), datetime.date(2017, 4, 16), datetime.date(2017, 5, 7), datetime.date(2017, 6, 25), datetime.date(2017, 7, 9), datetime.date(2017, 8, 20), datetime.date(2017, 9, 24), datetime.date(2017, 10, 29), datetime.date(2017, 11, 5), datetime.date(2017, 12, 31)]]
testing_days3 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 20), datetime.date(2017, 3, 6), datetime.date(2017, 4, 24), datetime.date(2017, 5, 8), datetime.date(2017, 6, 26), datetime.date(2017, 7, 17), datetime.date(2017, 8, 21), datetime.date(2017, 9, 11), datetime.date(2017, 10, 9), datetime.date(2017, 11, 6), datetime.date(2017, 12, 18)], [datetime.date(2017, 1, 17), datetime.date(2017, 2, 14), datetime.date(2017, 3, 21), datetime.date(2017, 4, 4), datetime.date(2017, 5, 2), datetime.date(2017, 6, 27), datetime.date(2017, 7, 4), datetime.date(2017, 8, 8), datetime.date(2017, 9, 5), datetime.date(2017, 10, 17), datetime.date(2017, 11, 7), datetime.date(2017, 12, 12)], [datetime.date(2017, 1, 25), datetime.date(2017, 2, 1), datetime.date(2017, 3, 8), datetime.date(2017, 4, 26), datetime.date(2017, 5, 10), datetime.date(2017, 6, 21), datetime.date(2017, 7, 26), datetime.date(2017, 8, 2), datetime.date(2017, 9, 6), datetime.date(2017, 10, 11), datetime.date(2017, 11, 8), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 2), datetime.date(2017, 3, 9), datetime.date(2017, 4, 20), datetime.date(2017, 5, 4), datetime.date(2017, 6, 8), datetime.date(2017, 7, 27), datetime.date(2017, 8, 3), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 16), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 13), datetime.date(2017, 2, 17), datetime.date(2017, 3, 31), datetime.date(2017, 4, 21), datetime.date(2017, 5, 19), datetime.date(2017, 6, 30), datetime.date(2017, 7, 21), datetime.date(2017, 8, 18), datetime.date(2017, 9, 8), datetime.date(2017, 10, 6), datetime.date(2017, 11, 3), datetime.date(2017, 12, 1)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 25), datetime.date(2017, 3, 11), datetime.date(2017, 4, 8), datetime.date(2017, 5, 20), datetime.date(2017, 6, 24), datetime.date(2017, 7, 15), datetime.date(2017, 8, 26), datetime.date(2017, 9, 30), datetime.date(2017, 10, 28), datetime.date(2017, 11, 4), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 5), datetime.date(2017, 3, 5), datetime.date(2017, 4, 2), datetime.date(2017, 5, 21), datetime.date(2017, 6, 4), datetime.date(2017, 7, 30), datetime.date(2017, 8, 6), datetime.date(2017, 9, 10), datetime.date(2017, 10, 8), datetime.date(2017, 11, 12), datetime.date(2017, 12, 24)]]
testing_days4 = [[datetime.date(2017, 1, 9), datetime.date(2017, 2, 20), datetime.date(2017, 3, 13), datetime.date(2017, 4, 10), datetime.date(2017, 5, 15), datetime.date(2017, 6, 12), datetime.date(2017, 7, 10), datetime.date(2017, 8, 21), datetime.date(2017, 9, 11), datetime.date(2017, 10, 30), datetime.date(2017, 11, 27), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 21), datetime.date(2017, 3, 21), datetime.date(2017, 4, 11), datetime.date(2017, 5, 2), datetime.date(2017, 6, 6), datetime.date(2017, 7, 25), datetime.date(2017, 8, 1), datetime.date(2017, 9, 19), datetime.date(2017, 10, 17), datetime.date(2017, 11, 21), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 25), datetime.date(2017, 2, 1), datetime.date(2017, 3, 29), datetime.date(2017, 4, 5), datetime.date(2017, 5, 3), datetime.date(2017, 6, 21), datetime.date(2017, 7, 26), datetime.date(2017, 8, 16), datetime.date(2017, 9, 6), datetime.date(2017, 10, 18), datetime.date(2017, 11, 15), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 2), datetime.date(2017, 3, 23), datetime.date(2017, 4, 6), datetime.date(2017, 5, 18), datetime.date(2017, 6, 29), datetime.date(2017, 7, 20), datetime.date(2017, 8, 17), datetime.date(2017, 9, 7), datetime.date(2017, 10, 12), datetime.date(2017, 11, 9), datetime.date(2017, 12, 7)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 10), datetime.date(2017, 3, 31), datetime.date(2017, 4, 21), datetime.date(2017, 5, 5), datetime.date(2017, 6, 23), datetime.date(2017, 7, 14), datetime.date(2017, 8, 25), datetime.date(2017, 9, 29), datetime.date(2017, 10, 13), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 18), datetime.date(2017, 3, 18), datetime.date(2017, 4, 22), datetime.date(2017, 5, 20), datetime.date(2017, 6, 17), datetime.date(2017, 7, 8), datetime.date(2017, 8, 26), datetime.date(2017, 9, 16), datetime.date(2017, 10, 21), datetime.date(2017, 11, 11), datetime.date(2017, 12, 30)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 26), datetime.date(2017, 3, 5), datetime.date(2017, 4, 23), datetime.date(2017, 5, 21), datetime.date(2017, 6, 18), datetime.date(2017, 7, 16), datetime.date(2017, 8, 20), datetime.date(2017, 9, 3), datetime.date(2017, 10, 8), datetime.date(2017, 11, 12), datetime.date(2017, 12, 3)]]
testing_days5 = [[datetime.date(2017, 1, 9), datetime.date(2017, 2, 20), datetime.date(2017, 3, 13), datetime.date(2017, 4, 3), datetime.date(2017, 5, 1), datetime.date(2017, 6, 5), datetime.date(2017, 7, 10), datetime.date(2017, 8, 7), datetime.date(2017, 9, 11), datetime.date(2017, 10, 16), datetime.date(2017, 11, 27), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 7), datetime.date(2017, 3, 7), datetime.date(2017, 4, 25), datetime.date(2017, 5, 16), datetime.date(2017, 6, 20), datetime.date(2017, 7, 4), datetime.date(2017, 8, 29), datetime.date(2017, 9, 12), datetime.date(2017, 10, 31), datetime.date(2017, 11, 28), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 15), datetime.date(2017, 3, 29), datetime.date(2017, 4, 12), datetime.date(2017, 5, 10), datetime.date(2017, 6, 7), datetime.date(2017, 7, 26), datetime.date(2017, 8, 2), datetime.date(2017, 9, 6), datetime.date(2017, 10, 25), datetime.date(2017, 11, 15), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 2), datetime.date(2017, 3, 2), datetime.date(2017, 4, 6), datetime.date(2017, 5, 18), datetime.date(2017, 6, 22), datetime.date(2017, 7, 13), datetime.date(2017, 8, 3), datetime.date(2017, 9, 28), datetime.date(2017, 10, 12), datetime.date(2017, 11, 30), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 3), datetime.date(2017, 3, 10), datetime.date(2017, 4, 21), datetime.date(2017, 5, 19), datetime.date(2017, 6, 16), datetime.date(2017, 7, 21), datetime.date(2017, 8, 4), datetime.date(2017, 9, 22), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 7), datetime.date(2017, 2, 4), datetime.date(2017, 3, 11), datetime.date(2017, 4, 22), datetime.date(2017, 5, 6), datetime.date(2017, 6, 10), datetime.date(2017, 7, 1), datetime.date(2017, 8, 12), datetime.date(2017, 9, 23), datetime.date(2017, 10, 21), datetime.date(2017, 11, 25), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 19), datetime.date(2017, 3, 19), datetime.date(2017, 4, 23), datetime.date(2017, 5, 7), datetime.date(2017, 6, 11), datetime.date(2017, 7, 30), datetime.date(2017, 8, 6), datetime.date(2017, 9, 3), datetime.date(2017, 10, 22), datetime.date(2017, 11, 5), datetime.date(2017, 12, 24)]]
testing_days6 = [[datetime.date(2017, 1, 2), datetime.date(2017, 2, 13), datetime.date(2017, 3, 27), datetime.date(2017, 4, 3), datetime.date(2017, 5, 8), datetime.date(2017, 6, 12), datetime.date(2017, 7, 24), datetime.date(2017, 8, 14), datetime.date(2017, 9, 25), datetime.date(2017, 10, 23), datetime.date(2017, 11, 27), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 28), datetime.date(2017, 3, 7), datetime.date(2017, 4, 25), datetime.date(2017, 5, 2), datetime.date(2017, 6, 6), datetime.date(2017, 7, 11), datetime.date(2017, 8, 1), datetime.date(2017, 9, 26), datetime.date(2017, 10, 17), datetime.date(2017, 11, 28), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 25), datetime.date(2017, 2, 8), datetime.date(2017, 3, 1), datetime.date(2017, 4, 12), datetime.date(2017, 5, 3), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 23), datetime.date(2017, 9, 20), datetime.date(2017, 10, 11), datetime.date(2017, 11, 1), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 9), datetime.date(2017, 3, 9), datetime.date(2017, 4, 13), datetime.date(2017, 5, 11), datetime.date(2017, 6, 29), datetime.date(2017, 7, 27), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 30), datetime.date(2017, 12, 7)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 3), datetime.date(2017, 3, 24), datetime.date(2017, 4, 21), datetime.date(2017, 5, 12), datetime.date(2017, 6, 30), datetime.date(2017, 7, 21), datetime.date(2017, 8, 25), datetime.date(2017, 9, 29), datetime.date(2017, 10, 6), datetime.date(2017, 11, 10), datetime.date(2017, 12, 22)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 11), datetime.date(2017, 3, 11), datetime.date(2017, 4, 8), datetime.date(2017, 5, 6), datetime.date(2017, 6, 17), datetime.date(2017, 7, 15), datetime.date(2017, 8, 26), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 18), datetime.date(2017, 12, 16)], [datetime.date(2017, 1, 29), datetime.date(2017, 2, 26), datetime.date(2017, 3, 12), datetime.date(2017, 4, 2), datetime.date(2017, 5, 7), datetime.date(2017, 6, 25), datetime.date(2017, 7, 2), datetime.date(2017, 8, 6), datetime.date(2017, 9, 24), datetime.date(2017, 10, 8), datetime.date(2017, 11, 26), datetime.date(2017, 12, 10)]]
testing_days7 = [[datetime.date(2017, 1, 9), datetime.date(2017, 2, 27), datetime.date(2017, 3, 6), datetime.date(2017, 4, 17), datetime.date(2017, 5, 8), datetime.date(2017, 6, 26), datetime.date(2017, 7, 24), datetime.date(2017, 8, 28), datetime.date(2017, 9, 11), datetime.date(2017, 10, 16), datetime.date(2017, 11, 27), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 7), datetime.date(2017, 3, 7), datetime.date(2017, 4, 11), datetime.date(2017, 5, 2), datetime.date(2017, 6, 20), datetime.date(2017, 7, 18), datetime.date(2017, 8, 22), datetime.date(2017, 9, 12), datetime.date(2017, 10, 10), datetime.date(2017, 11, 21), datetime.date(2017, 12, 12)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 8), datetime.date(2017, 3, 29), datetime.date(2017, 4, 5), datetime.date(2017, 5, 17), datetime.date(2017, 6, 14), datetime.date(2017, 7, 19), datetime.date(2017, 8, 2), datetime.date(2017, 9, 13), datetime.date(2017, 10, 4), datetime.date(2017, 11, 22), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 23), datetime.date(2017, 3, 23), datetime.date(2017, 4, 27), datetime.date(2017, 5, 4), datetime.date(2017, 6, 1), datetime.date(2017, 7, 6), datetime.date(2017, 8, 24), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 9), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 17), datetime.date(2017, 3, 17), datetime.date(2017, 4, 14), datetime.date(2017, 5, 12), datetime.date(2017, 6, 23), datetime.date(2017, 7, 28), datetime.date(2017, 8, 18), datetime.date(2017, 9, 1), datetime.date(2017, 10, 27), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 11), datetime.date(2017, 3, 25), datetime.date(2017, 4, 29), datetime.date(2017, 5, 6), datetime.date(2017, 6, 24), datetime.date(2017, 7, 15), datetime.date(2017, 8, 5), datetime.date(2017, 9, 2), datetime.date(2017, 10, 7), datetime.date(2017, 11, 25), datetime.date(2017, 12, 2)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 19), datetime.date(2017, 3, 5), datetime.date(2017, 4, 2), datetime.date(2017, 5, 14), datetime.date(2017, 6, 11), datetime.date(2017, 7, 30), datetime.date(2017, 8, 20), datetime.date(2017, 9, 10), datetime.date(2017, 10, 1), datetime.date(2017, 11, 19), datetime.date(2017, 12, 17)]]
testing_days8 = [[datetime.date(2017, 1, 16), datetime.date(2017, 2, 13), datetime.date(2017, 3, 20), datetime.date(2017, 4, 3), datetime.date(2017, 5, 1), datetime.date(2017, 6, 19), datetime.date(2017, 7, 24), datetime.date(2017, 8, 7), datetime.date(2017, 9, 4), datetime.date(2017, 10, 9), datetime.date(2017, 11, 6), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 10), datetime.date(2017, 2, 14), datetime.date(2017, 3, 7), datetime.date(2017, 4, 18), datetime.date(2017, 5, 9), datetime.date(2017, 6, 13), datetime.date(2017, 7, 25), datetime.date(2017, 8, 22), datetime.date(2017, 9, 5), datetime.date(2017, 10, 31), datetime.date(2017, 11, 7), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 1), datetime.date(2017, 3, 1), datetime.date(2017, 4, 5), datetime.date(2017, 5, 10), datetime.date(2017, 6, 7), datetime.date(2017, 7, 26), datetime.date(2017, 8, 9), datetime.date(2017, 9, 6), datetime.date(2017, 10, 11), datetime.date(2017, 11, 29), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 23), datetime.date(2017, 3, 16), datetime.date(2017, 4, 13), datetime.date(2017, 5, 11), datetime.date(2017, 6, 8), datetime.date(2017, 7, 20), datetime.date(2017, 8, 24), datetime.date(2017, 9, 21), datetime.date(2017, 10, 5), datetime.date(2017, 11, 30), datetime.date(2017, 12, 7)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 10), datetime.date(2017, 3, 24), datetime.date(2017, 4, 21), datetime.date(2017, 5, 12), datetime.date(2017, 6, 16), datetime.date(2017, 7, 7), datetime.date(2017, 8, 18), datetime.date(2017, 9, 22), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 18), datetime.date(2017, 3, 4), datetime.date(2017, 4, 22), datetime.date(2017, 5, 20), datetime.date(2017, 6, 24), datetime.date(2017, 7, 22), datetime.date(2017, 8, 26), datetime.date(2017, 9, 30), datetime.date(2017, 10, 21), datetime.date(2017, 11, 4), datetime.date(2017, 12, 9)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 5), datetime.date(2017, 3, 19), datetime.date(2017, 4, 23), datetime.date(2017, 5, 21), datetime.date(2017, 6, 11), datetime.date(2017, 7, 30), datetime.date(2017, 8, 13), datetime.date(2017, 9, 3), datetime.date(2017, 10, 8), datetime.date(2017, 11, 26), datetime.date(2017, 12, 3)]]
testing_days9 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 13), datetime.date(2017, 3, 13), datetime.date(2017, 4, 10), datetime.date(2017, 5, 15), datetime.date(2017, 6, 12), datetime.date(2017, 7, 17), datetime.date(2017, 8, 28), datetime.date(2017, 9, 18), datetime.date(2017, 10, 2), datetime.date(2017, 11, 20), datetime.date(2017, 12, 18)], [datetime.date(2017, 1, 10), datetime.date(2017, 2, 14), datetime.date(2017, 3, 7), datetime.date(2017, 4, 18), datetime.date(2017, 5, 16), datetime.date(2017, 6, 13), datetime.date(2017, 7, 4), datetime.date(2017, 8, 29), datetime.date(2017, 9, 12), datetime.date(2017, 10, 24), datetime.date(2017, 11, 14), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 8), datetime.date(2017, 3, 1), datetime.date(2017, 4, 26), datetime.date(2017, 5, 17), datetime.date(2017, 6, 14), datetime.date(2017, 7, 5), datetime.date(2017, 8, 16), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 8), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 9), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 1), datetime.date(2017, 7, 20), datetime.date(2017, 8, 3), datetime.date(2017, 9, 21), datetime.date(2017, 10, 12), datetime.date(2017, 11, 30), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 27), datetime.date(2017, 2, 3), datetime.date(2017, 3, 31), datetime.date(2017, 4, 21), datetime.date(2017, 5, 19), datetime.date(2017, 6, 16), datetime.date(2017, 7, 14), datetime.date(2017, 8, 4), datetime.date(2017, 9, 8), datetime.date(2017, 10, 13), datetime.date(2017, 11, 17), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 25), datetime.date(2017, 3, 25), datetime.date(2017, 4, 22), datetime.date(2017, 5, 6), datetime.date(2017, 6, 3), datetime.date(2017, 7, 29), datetime.date(2017, 8, 12), datetime.date(2017, 9, 9), datetime.date(2017, 10, 21), datetime.date(2017, 11, 18), datetime.date(2017, 12, 30)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 12), datetime.date(2017, 3, 19), datetime.date(2017, 4, 9), datetime.date(2017, 5, 14), datetime.date(2017, 6, 11), datetime.date(2017, 7, 23), datetime.date(2017, 8, 20), datetime.date(2017, 9, 17), datetime.date(2017, 10, 22), datetime.date(2017, 11, 26), datetime.date(2017, 12, 10)]]
testing_days10 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 6), datetime.date(2017, 3, 27), datetime.date(2017, 4, 24), datetime.date(2017, 5, 1), datetime.date(2017, 6, 26), datetime.date(2017, 7, 10), datetime.date(2017, 8, 7), datetime.date(2017, 9, 18), datetime.date(2017, 10, 16), datetime.date(2017, 11, 27), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 28), datetime.date(2017, 3, 21), datetime.date(2017, 4, 18), datetime.date(2017, 5, 9), datetime.date(2017, 6, 13), datetime.date(2017, 7, 11), datetime.date(2017, 8, 1), datetime.date(2017, 9, 5), datetime.date(2017, 10, 17), datetime.date(2017, 11, 28), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 22), datetime.date(2017, 3, 22), datetime.date(2017, 4, 26), datetime.date(2017, 5, 17), datetime.date(2017, 6, 28), datetime.date(2017, 7, 5), datetime.date(2017, 8, 2), datetime.date(2017, 9, 6), datetime.date(2017, 10, 4), datetime.date(2017, 11, 8), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 16), datetime.date(2017, 3, 9), datetime.date(2017, 4, 20), datetime.date(2017, 5, 18), datetime.date(2017, 6, 22), datetime.date(2017, 7, 6), datetime.date(2017, 8, 24), datetime.date(2017, 9, 21), datetime.date(2017, 10, 26), datetime.date(2017, 11, 9), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 17), datetime.date(2017, 3, 24), datetime.date(2017, 4, 28), datetime.date(2017, 5, 12), datetime.date(2017, 6, 2), datetime.date(2017, 7, 28), datetime.date(2017, 8, 18), datetime.date(2017, 9, 22), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 7), datetime.date(2017, 2, 11), datetime.date(2017, 3, 18), datetime.date(2017, 4, 1), datetime.date(2017, 5, 13), datetime.date(2017, 6, 3), datetime.date(2017, 7, 22), datetime.date(2017, 8, 19), datetime.date(2017, 9, 23), datetime.date(2017, 10, 28), datetime.date(2017, 11, 4), datetime.date(2017, 12, 9)], [datetime.date(2017, 1, 8), datetime.date(2017, 2, 19), datetime.date(2017, 3, 26), datetime.date(2017, 4, 23), datetime.date(2017, 5, 7), datetime.date(2017, 6, 11), datetime.date(2017, 7, 9), datetime.date(2017, 8, 13), datetime.date(2017, 9, 10), datetime.date(2017, 10, 29), datetime.date(2017, 11, 26), datetime.date(2017, 12, 3)]]
testing_days11 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 20), datetime.date(2017, 3, 13), datetime.date(2017, 4, 24), datetime.date(2017, 5, 1), datetime.date(2017, 6, 12), datetime.date(2017, 7, 24), datetime.date(2017, 8, 7), datetime.date(2017, 9, 18), datetime.date(2017, 10, 23), datetime.date(2017, 11, 6), datetime.date(2017, 12, 4)], [datetime.date(2017, 1, 10), datetime.date(2017, 2, 28), datetime.date(2017, 3, 7), datetime.date(2017, 4, 4), datetime.date(2017, 5, 2), datetime.date(2017, 6, 13), datetime.date(2017, 7, 25), datetime.date(2017, 8, 8), datetime.date(2017, 9, 5), datetime.date(2017, 10, 24), datetime.date(2017, 11, 28), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 22), datetime.date(2017, 3, 15), datetime.date(2017, 4, 19), datetime.date(2017, 5, 17), datetime.date(2017, 6, 21), datetime.date(2017, 7, 12), datetime.date(2017, 8, 23), datetime.date(2017, 9, 27), datetime.date(2017, 10, 11), datetime.date(2017, 11, 15), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 9), datetime.date(2017, 3, 23), datetime.date(2017, 4, 13), datetime.date(2017, 5, 18), datetime.date(2017, 6, 22), datetime.date(2017, 7, 6), datetime.date(2017, 8, 3), datetime.date(2017, 9, 14), datetime.date(2017, 10, 5), datetime.date(2017, 11, 2), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 13), datetime.date(2017, 2, 10), datetime.date(2017, 3, 31), datetime.date(2017, 4, 28), datetime.date(2017, 5, 5), datetime.date(2017, 6, 16), datetime.date(2017, 7, 14), datetime.date(2017, 8, 4), datetime.date(2017, 9, 15), datetime.date(2017, 10, 27), datetime.date(2017, 11, 17), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 28), datetime.date(2017, 2, 25), datetime.date(2017, 3, 4), datetime.date(2017, 4, 8), datetime.date(2017, 5, 13), datetime.date(2017, 6, 10), datetime.date(2017, 7, 15), datetime.date(2017, 8, 12), datetime.date(2017, 9, 9), datetime.date(2017, 10, 14), datetime.date(2017, 11, 25), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 12), datetime.date(2017, 3, 19), datetime.date(2017, 4, 9), datetime.date(2017, 5, 21), datetime.date(2017, 6, 25), datetime.date(2017, 7, 23), datetime.date(2017, 8, 13), datetime.date(2017, 9, 17), datetime.date(2017, 10, 29), datetime.date(2017, 11, 5), datetime.date(2017, 12, 31)]]
testing_days12 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 17), datetime.date(2017, 5, 8), datetime.date(2017, 6, 19), datetime.date(2017, 7, 31), datetime.date(2017, 8, 14), datetime.date(2017, 9, 4), datetime.date(2017, 10, 30), datetime.date(2017, 11, 27), datetime.date(2017, 12, 18)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 7), datetime.date(2017, 3, 7), datetime.date(2017, 4, 4), datetime.date(2017, 5, 16), datetime.date(2017, 6, 13), datetime.date(2017, 7, 11), datetime.date(2017, 8, 29), datetime.date(2017, 9, 12), datetime.date(2017, 10, 31), datetime.date(2017, 11, 7), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 22), datetime.date(2017, 3, 22), datetime.date(2017, 4, 19), datetime.date(2017, 5, 17), datetime.date(2017, 6, 28), datetime.date(2017, 7, 12), datetime.date(2017, 8, 23), datetime.date(2017, 9, 6), datetime.date(2017, 10, 11), datetime.date(2017, 11, 22), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 16), datetime.date(2017, 3, 2), datetime.date(2017, 4, 6), datetime.date(2017, 5, 18), datetime.date(2017, 6, 15), datetime.date(2017, 7, 13), datetime.date(2017, 8, 17), datetime.date(2017, 9, 7), datetime.date(2017, 10, 5), datetime.date(2017, 11, 2), datetime.date(2017, 12, 14)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 10), datetime.date(2017, 3, 3), datetime.date(2017, 4, 7), datetime.date(2017, 5, 5), datetime.date(2017, 6, 9), datetime.date(2017, 7, 14), datetime.date(2017, 8, 4), datetime.date(2017, 9, 22), datetime.date(2017, 10, 6), datetime.date(2017, 11, 24), datetime.date(2017, 12, 8)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 25), datetime.date(2017, 3, 4), datetime.date(2017, 4, 29), datetime.date(2017, 5, 6), datetime.date(2017, 6, 24), datetime.date(2017, 7, 1), datetime.date(2017, 8, 26), datetime.date(2017, 9, 23), datetime.date(2017, 10, 21), datetime.date(2017, 11, 4), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 19), datetime.date(2017, 3, 19), datetime.date(2017, 4, 23), datetime.date(2017, 5, 21), datetime.date(2017, 6, 11), datetime.date(2017, 7, 23), datetime.date(2017, 8, 20), datetime.date(2017, 9, 10), datetime.date(2017, 10, 8), datetime.date(2017, 11, 26), datetime.date(2017, 12, 17)]]
testing_days13 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 17), datetime.date(2017, 5, 1), datetime.date(2017, 6, 19), datetime.date(2017, 7, 24), datetime.date(2017, 8, 21), datetime.date(2017, 9, 25), datetime.date(2017, 10, 30), datetime.date(2017, 11, 27), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 14), datetime.date(2017, 3, 21), datetime.date(2017, 4, 11), datetime.date(2017, 5, 9), datetime.date(2017, 6, 27), datetime.date(2017, 7, 25), datetime.date(2017, 8, 15), datetime.date(2017, 9, 5), datetime.date(2017, 10, 10), datetime.date(2017, 11, 7), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 1), datetime.date(2017, 3, 15), datetime.date(2017, 4, 5), datetime.date(2017, 5, 17), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 16), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 29), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 2), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 22), datetime.date(2017, 7, 20), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 5), datetime.date(2017, 11, 16), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 10), datetime.date(2017, 3, 3), datetime.date(2017, 4, 7), datetime.date(2017, 5, 5), datetime.date(2017, 6, 30), datetime.date(2017, 7, 7), datetime.date(2017, 8, 11), datetime.date(2017, 9, 8), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 28), datetime.date(2017, 2, 11), datetime.date(2017, 3, 18), datetime.date(2017, 4, 1), datetime.date(2017, 5, 6), datetime.date(2017, 6, 17), datetime.date(2017, 7, 8), datetime.date(2017, 8, 26), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 4), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 19), datetime.date(2017, 3, 5), datetime.date(2017, 4, 9), datetime.date(2017, 5, 14), datetime.date(2017, 6, 4), datetime.date(2017, 7, 9), datetime.date(2017, 8, 13), datetime.date(2017, 9, 10), datetime.date(2017, 10, 22), datetime.date(2017, 11, 5), datetime.date(2017, 12, 31)]]
testing_days14 = [[datetime.date(2017, 1, 16), datetime.date(2017, 2, 27), datetime.date(2017, 3, 27), datetime.date(2017, 4, 24), datetime.date(2017, 5, 8), datetime.date(2017, 6, 5), datetime.date(2017, 7, 3), datetime.date(2017, 8, 28), datetime.date(2017, 9, 25), datetime.date(2017, 10, 9), datetime.date(2017, 11, 13), datetime.date(2017, 12, 4)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 28), datetime.date(2017, 3, 28), datetime.date(2017, 4, 4), datetime.date(2017, 5, 16), datetime.date(2017, 6, 13), datetime.date(2017, 7, 18), datetime.date(2017, 8, 29), datetime.date(2017, 9, 5), datetime.date(2017, 10, 17), datetime.date(2017, 11, 7), datetime.date(2017, 12, 12)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 8), datetime.date(2017, 3, 1), datetime.date(2017, 4, 19), datetime.date(2017, 5, 17), datetime.date(2017, 6, 7), datetime.date(2017, 7, 5), datetime.date(2017, 8, 23), datetime.date(2017, 9, 27), datetime.date(2017, 10, 4), datetime.date(2017, 11, 22), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 16), datetime.date(2017, 3, 23), datetime.date(2017, 4, 27), datetime.date(2017, 5, 11), datetime.date(2017, 6, 15), datetime.date(2017, 7, 13), datetime.date(2017, 8, 24), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 16), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 13), datetime.date(2017, 2, 10), datetime.date(2017, 3, 10), datetime.date(2017, 4, 21), datetime.date(2017, 5, 12), datetime.date(2017, 6, 9), datetime.date(2017, 7, 7), datetime.date(2017, 8, 11), datetime.date(2017, 9, 15), datetime.date(2017, 10, 20), datetime.date(2017, 11, 17), datetime.date(2017, 12, 22)], [datetime.date(2017, 1, 7), datetime.date(2017, 2, 11), datetime.date(2017, 3, 4), datetime.date(2017, 4, 8), datetime.date(2017, 5, 13), datetime.date(2017, 6, 17), datetime.date(2017, 7, 15), datetime.date(2017, 8, 5), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 4), datetime.date(2017, 12, 2)], [datetime.date(2017, 1, 15), datetime.date(2017, 2, 5), datetime.date(2017, 3, 5), datetime.date(2017, 4, 2), datetime.date(2017, 5, 7), datetime.date(2017, 6, 11), datetime.date(2017, 7, 16), datetime.date(2017, 8, 13), datetime.date(2017, 9, 17), datetime.date(2017, 10, 22), datetime.date(2017, 11, 19), datetime.date(2017, 12, 3)]]
testing_days15 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 10), datetime.date(2017, 5, 1), datetime.date(2017, 6, 5), datetime.date(2017, 7, 31), datetime.date(2017, 8, 7), datetime.date(2017, 9, 4), datetime.date(2017, 10, 30), datetime.date(2017, 11, 6), datetime.date(2017, 12, 4)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 28), datetime.date(2017, 3, 14), datetime.date(2017, 4, 18), datetime.date(2017, 5, 16), datetime.date(2017, 6, 27), datetime.date(2017, 7, 11), datetime.date(2017, 8, 22), datetime.date(2017, 9, 12), datetime.date(2017, 10, 17), datetime.date(2017, 11, 14), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 25), datetime.date(2017, 2, 8), datetime.date(2017, 3, 22), datetime.date(2017, 4, 26), datetime.date(2017, 5, 17), datetime.date(2017, 6, 14), datetime.date(2017, 7, 5), datetime.date(2017, 8, 9), datetime.date(2017, 9, 6), datetime.date(2017, 10, 18), datetime.date(2017, 11, 8), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 9), datetime.date(2017, 3, 30), datetime.date(2017, 4, 6), datetime.date(2017, 5, 18), datetime.date(2017, 6, 1), datetime.date(2017, 7, 13), datetime.date(2017, 8, 10), datetime.date(2017, 9, 21), datetime.date(2017, 10, 26), datetime.date(2017, 11, 9), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 13), datetime.date(2017, 2, 24), datetime.date(2017, 3, 31), datetime.date(2017, 4, 7), datetime.date(2017, 5, 12), datetime.date(2017, 6, 16), datetime.date(2017, 7, 7), datetime.date(2017, 8, 4), datetime.date(2017, 9, 22), datetime.date(2017, 10, 27), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 28), datetime.date(2017, 2, 25), datetime.date(2017, 3, 25), datetime.date(2017, 4, 29), datetime.date(2017, 5, 20), datetime.date(2017, 6, 10), datetime.date(2017, 7, 22), datetime.date(2017, 8, 26), datetime.date(2017, 9, 2), datetime.date(2017, 10, 7), datetime.date(2017, 11, 11), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 5), datetime.date(2017, 3, 5), datetime.date(2017, 4, 2), datetime.date(2017, 5, 7), datetime.date(2017, 6, 25), datetime.date(2017, 7, 23), datetime.date(2017, 8, 20), datetime.date(2017, 9, 24), datetime.date(2017, 10, 15), datetime.date(2017, 11, 5), datetime.date(2017, 12, 17)]]
testing_days16 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 20), datetime.date(2017, 3, 13), datetime.date(2017, 4, 24), datetime.date(2017, 5, 15), datetime.date(2017, 6, 12), datetime.date(2017, 7, 24), datetime.date(2017, 8, 28), datetime.date(2017, 9, 11), datetime.date(2017, 10, 23), datetime.date(2017, 11, 13), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 31), datetime.date(2017, 2, 21), datetime.date(2017, 3, 21), datetime.date(2017, 4, 4), datetime.date(2017, 5, 16), datetime.date(2017, 6, 20), datetime.date(2017, 7, 18), datetime.date(2017, 8, 29), datetime.date(2017, 9, 12), datetime.date(2017, 10, 31), datetime.date(2017, 11, 14), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 15), datetime.date(2017, 3, 15), datetime.date(2017, 4, 26), datetime.date(2017, 5, 3), datetime.date(2017, 6, 7), datetime.date(2017, 7, 12), datetime.date(2017, 8, 16), datetime.date(2017, 9, 6), datetime.date(2017, 10, 4), datetime.date(2017, 11, 8), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 26), datetime.date(2017, 2, 9), datetime.date(2017, 3, 16), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 22), datetime.date(2017, 7, 20), datetime.date(2017, 8, 3), datetime.date(2017, 9, 7), datetime.date(2017, 10, 12), datetime.date(2017, 11, 2), datetime.date(2017, 12, 7)], [datetime.date(2017, 1, 27), datetime.date(2017, 2, 24), datetime.date(2017, 3, 31), datetime.date(2017, 4, 14), datetime.date(2017, 5, 12), datetime.date(2017, 6, 30), datetime.date(2017, 7, 28), datetime.date(2017, 8, 25), datetime.date(2017, 9, 29), datetime.date(2017, 10, 20), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 25), datetime.date(2017, 3, 4), datetime.date(2017, 4, 22), datetime.date(2017, 5, 6), datetime.date(2017, 6, 17), datetime.date(2017, 7, 29), datetime.date(2017, 8, 19), datetime.date(2017, 9, 9), datetime.date(2017, 10, 14), datetime.date(2017, 11, 25), datetime.date(2017, 12, 30)], [datetime.date(2017, 1, 8), datetime.date(2017, 2, 5), datetime.date(2017, 3, 19), datetime.date(2017, 4, 9), datetime.date(2017, 5, 7), datetime.date(2017, 6, 4), datetime.date(2017, 7, 2), datetime.date(2017, 8, 6), datetime.date(2017, 9, 3), datetime.date(2017, 10, 8), datetime.date(2017, 11, 26), datetime.date(2017, 12, 24)]]
testing_days17 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 6), datetime.date(2017, 3, 6), datetime.date(2017, 4, 10), datetime.date(2017, 5, 1), datetime.date(2017, 6, 5), datetime.date(2017, 7, 17), datetime.date(2017, 8, 28), datetime.date(2017, 9, 11), datetime.date(2017, 10, 9), datetime.date(2017, 11, 6), datetime.date(2017, 12, 18)], [datetime.date(2017, 1, 31), datetime.date(2017, 2, 14), datetime.date(2017, 3, 28), datetime.date(2017, 4, 18), datetime.date(2017, 5, 2), datetime.date(2017, 6, 27), datetime.date(2017, 7, 25), datetime.date(2017, 8, 15), datetime.date(2017, 9, 26), datetime.date(2017, 10, 17), datetime.date(2017, 11, 7), datetime.date(2017, 12, 19)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 22), datetime.date(2017, 3, 22), datetime.date(2017, 4, 26), datetime.date(2017, 5, 10), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 2), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 22), datetime.date(2017, 12, 6)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 16), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 1), datetime.date(2017, 7, 27), datetime.date(2017, 8, 3), datetime.date(2017, 9, 7), datetime.date(2017, 10, 26), datetime.date(2017, 11, 9), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 3), datetime.date(2017, 3, 17), datetime.date(2017, 4, 14), datetime.date(2017, 5, 5), datetime.date(2017, 6, 9), datetime.date(2017, 7, 7), datetime.date(2017, 8, 25), datetime.date(2017, 9, 8), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 22)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 18), datetime.date(2017, 3, 4), datetime.date(2017, 4, 22), datetime.date(2017, 5, 6), datetime.date(2017, 6, 24), datetime.date(2017, 7, 8), datetime.date(2017, 8, 19), datetime.date(2017, 9, 9), datetime.date(2017, 10, 7), datetime.date(2017, 11, 11), datetime.date(2017, 12, 16)], [datetime.date(2017, 1, 29), datetime.date(2017, 2, 5), datetime.date(2017, 3, 26), datetime.date(2017, 4, 23), datetime.date(2017, 5, 7), datetime.date(2017, 6, 18), datetime.date(2017, 7, 9), datetime.date(2017, 8, 20), datetime.date(2017, 9, 10), datetime.date(2017, 10, 15), datetime.date(2017, 11, 5), datetime.date(2017, 12, 17)]]
testing_days18 = [[datetime.date(2017, 1, 2), datetime.date(2017, 2, 20), datetime.date(2017, 3, 27), datetime.date(2017, 4, 17), datetime.date(2017, 5, 15), datetime.date(2017, 6, 12), datetime.date(2017, 7, 17), datetime.date(2017, 8, 7), datetime.date(2017, 9, 25), datetime.date(2017, 10, 9), datetime.date(2017, 11, 20), datetime.date(2017, 12, 4)], [datetime.date(2017, 1, 17), datetime.date(2017, 2, 28), datetime.date(2017, 3, 28), datetime.date(2017, 4, 11), datetime.date(2017, 5, 16), datetime.date(2017, 6, 13), datetime.date(2017, 7, 25), datetime.date(2017, 8, 1), datetime.date(2017, 9, 12), datetime.date(2017, 10, 31), datetime.date(2017, 11, 21), datetime.date(2017, 12, 26)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 1), datetime.date(2017, 3, 22), datetime.date(2017, 4, 19), datetime.date(2017, 5, 17), datetime.date(2017, 6, 21), datetime.date(2017, 7, 26), datetime.date(2017, 8, 2), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 8), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 23), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 18), datetime.date(2017, 6, 8), datetime.date(2017, 7, 27), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 23), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 13), datetime.date(2017, 2, 10), datetime.date(2017, 3, 3), datetime.date(2017, 4, 28), datetime.date(2017, 5, 12), datetime.date(2017, 6, 9), datetime.date(2017, 7, 7), datetime.date(2017, 8, 25), datetime.date(2017, 9, 15), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 28), datetime.date(2017, 2, 25), datetime.date(2017, 3, 18), datetime.date(2017, 4, 22), datetime.date(2017, 5, 20), datetime.date(2017, 6, 24), datetime.date(2017, 7, 22), datetime.date(2017, 8, 5), datetime.date(2017, 9, 30), datetime.date(2017, 10, 28), datetime.date(2017, 11, 11), datetime.date(2017, 12, 16)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 12), datetime.date(2017, 3, 26), datetime.date(2017, 4, 9), datetime.date(2017, 5, 14), datetime.date(2017, 6, 18), datetime.date(2017, 7, 9), datetime.date(2017, 8, 13), datetime.date(2017, 9, 3), datetime.date(2017, 10, 15), datetime.date(2017, 11, 26), datetime.date(2017, 12, 24)]]
testing_days19 = [[datetime.date(2017, 1, 23), datetime.date(2017, 2, 6), datetime.date(2017, 3, 20), datetime.date(2017, 4, 10), datetime.date(2017, 5, 15), datetime.date(2017, 6, 19), datetime.date(2017, 7, 31), datetime.date(2017, 8, 28), datetime.date(2017, 9, 25), datetime.date(2017, 10, 16), datetime.date(2017, 11, 6), datetime.date(2017, 12, 18)], [datetime.date(2017, 1, 10), datetime.date(2017, 2, 21), datetime.date(2017, 3, 14), datetime.date(2017, 4, 25), datetime.date(2017, 5, 16), datetime.date(2017, 6, 27), datetime.date(2017, 7, 11), datetime.date(2017, 8, 22), datetime.date(2017, 9, 19), datetime.date(2017, 10, 10), datetime.date(2017, 11, 7), datetime.date(2017, 12, 26)], [datetime.date(2017, 1, 18), datetime.date(2017, 2, 8), datetime.date(2017, 3, 29), datetime.date(2017, 4, 19), datetime.date(2017, 5, 3), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 23), datetime.date(2017, 9, 27), datetime.date(2017, 10, 4), datetime.date(2017, 11, 22), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 9), datetime.date(2017, 3, 16), datetime.date(2017, 4, 6), datetime.date(2017, 5, 4), datetime.date(2017, 6, 22), datetime.date(2017, 7, 27), datetime.date(2017, 8, 3), datetime.date(2017, 9, 28), datetime.date(2017, 10, 12), datetime.date(2017, 11, 16), datetime.date(2017, 12, 28)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 3), datetime.date(2017, 3, 3), datetime.date(2017, 4, 14), datetime.date(2017, 5, 19), datetime.date(2017, 6, 23), datetime.date(2017, 7, 28), datetime.date(2017, 8, 25), datetime.date(2017, 9, 1), datetime.date(2017, 10, 13), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 21), datetime.date(2017, 2, 25), datetime.date(2017, 3, 18), datetime.date(2017, 4, 29), datetime.date(2017, 5, 6), datetime.date(2017, 6, 3), datetime.date(2017, 7, 1), datetime.date(2017, 8, 19), datetime.date(2017, 9, 16), datetime.date(2017, 10, 7), datetime.date(2017, 11, 25), datetime.date(2017, 12, 16)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 5), datetime.date(2017, 3, 12), datetime.date(2017, 4, 23), datetime.date(2017, 5, 7), datetime.date(2017, 6, 11), datetime.date(2017, 7, 9), datetime.date(2017, 8, 27), datetime.date(2017, 9, 24), datetime.date(2017, 10, 8), datetime.date(2017, 11, 5), datetime.date(2017, 12, 17)]]
testing_days20 = [[datetime.date(2017, 1, 9), datetime.date(2017, 2, 13), datetime.date(2017, 3, 6), datetime.date(2017, 4, 3), datetime.date(2017, 5, 15), datetime.date(2017, 6, 19), datetime.date(2017, 7, 3), datetime.date(2017, 8, 14), datetime.date(2017, 9, 4), datetime.date(2017, 10, 9), datetime.date(2017, 11, 6), datetime.date(2017, 12, 11)], [datetime.date(2017, 1, 3), datetime.date(2017, 2, 14), datetime.date(2017, 3, 28), datetime.date(2017, 4, 18), datetime.date(2017, 5, 16), datetime.date(2017, 6, 20), datetime.date(2017, 7, 4), datetime.date(2017, 8, 22), datetime.date(2017, 9, 5), datetime.date(2017, 10, 31), datetime.date(2017, 11, 7), datetime.date(2017, 12, 26)], [datetime.date(2017, 1, 25), datetime.date(2017, 2, 8), datetime.date(2017, 3, 8), datetime.date(2017, 4, 26), datetime.date(2017, 5, 10), datetime.date(2017, 6, 28), datetime.date(2017, 7, 5), datetime.date(2017, 8, 9), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 8), datetime.date(2017, 12, 13)], [datetime.date(2017, 1, 12), datetime.date(2017, 2, 9), datetime.date(2017, 3, 2), datetime.date(2017, 4, 13), datetime.date(2017, 5, 11), datetime.date(2017, 6, 8), datetime.date(2017, 7, 20), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 26), datetime.date(2017, 11, 2), datetime.date(2017, 12, 14)], [datetime.date(2017, 1, 6), datetime.date(2017, 2, 10), datetime.date(2017, 3, 31), datetime.date(2017, 4, 21), datetime.date(2017, 5, 19), datetime.date(2017, 6, 23), datetime.date(2017, 7, 14), datetime.date(2017, 8, 25), datetime.date(2017, 9, 8), datetime.date(2017, 10, 27), datetime.date(2017, 11, 10), datetime.date(2017, 12, 15)], [datetime.date(2017, 1, 14), datetime.date(2017, 2, 25), datetime.date(2017, 3, 4), datetime.date(2017, 4, 15), datetime.date(2017, 5, 6), datetime.date(2017, 6, 3), datetime.date(2017, 7, 1), datetime.date(2017, 8, 19), datetime.date(2017, 9, 30), datetime.date(2017, 10, 7), datetime.date(2017, 11, 18), datetime.date(2017, 12, 9)], [datetime.date(2017, 1, 8), datetime.date(2017, 2, 26), datetime.date(2017, 3, 5), datetime.date(2017, 4, 23), datetime.date(2017, 5, 7), datetime.date(2017, 6, 11), datetime.date(2017, 7, 2), datetime.date(2017, 8, 6), datetime.date(2017, 9, 3), datetime.date(2017, 10, 1), datetime.date(2017, 11, 12), datetime.date(2017, 12, 31)]]

date_exclude=[datetime.date(2017,2,21),datetime.date(2017,1,4),datetime.date(2017,1,11),datetime.date(2017,1,25), datetime.date(2017,2,6), datetime.date(2017,5,17), datetime.date(2017,9,27),datetime.date(2017,1,19), datetime.date(2017,10,19), datetime.date(2017,1,26)]
all_date=["20171005","20170102", "20170109", "20170116", "20170123", "20170130", "20170206", "20170213", "20170220", "20170227", "20170306", "20170313", "20170320", "20170327", "20170403", "20170410", "20170417", "20170424", "20170501", "20170508", "20170515", "20170522", "20170529", "20170605", "20170612", "20170619", "20170626", "20170703", "20170710", "20170717", "20170724", "20170731", "20170807", "20170814", "20170821", "20170828", "20170904", "20170911", "20170918", "20170925", "20171002", "20171009", "20171016", "20171023", "20171030", "20171106", "20171113", "20171120", "20171127", "20171204", "20171211", "20171218", "20171225", "20170102", "20170109", "20170116", "20170123", "20170130", "20170206", "20170213", "20170220", "20170227", "20170306", "20170313", "20170320", "20170327", "20170403", "20170410", "20170417", "20170424", "20170501", "20170508", "20170515", "20170522", "20170529", "20170605", "20170612", "20170619", "20170626", "20170703", "20170710", "20170717", "20170724", "20170731", "20170807", "20170814", "20170821", "20170828", "20170904", "20170911", "20170918", "20170925", "20171002", "20171009", "20171016", "20171023", "20171030", "20171106", "20171113", "20171120", "20171127", "20171204", "20171211", "20171218", "20171225", "20170103", "20170110", "20170117", "20170124", "20170131", "20170207", "20170214", "20170221", "20170228", "20170307", "20170314", "20170321", "20170328", "20170404", "20170411", "20170418", "20170425", "20170502", "20170509", "20170516", "20170523", "20170530", "20170606", "20170613", "20170620", "20170627", "20170704", "20170711", "20170718", "20170725", "20170801", "20170808", "20170815", "20170822", "20170829", "20170905", "20170912", "20170919", "20170926", "20171003", "20171010", "20171017", "20171024", "20171031", "20171107", "20171114", "20171121", "20171128", "20171205", "20171212", "20171219", "20171226", "20170104", "20170111", "20170118", "20170125", "20170201", "20170208", "20170215", "20170222", "20170301", "20170308", "20170315", "20170322", "20170329", "20170405", "20170412", "20170419", "20170426", "20170503", "20170510", "20170517", "20170524", "20170531", "20170607", "20170614", "20170621", "20170628", "20170705", "20170712", "20170719", "20170726", "20170802", "20170809", "20170816", "20170823", "20170830", "20170906", "20170913", "20170920", "20170927", "20171004", "20171011", "20171018", "20171025", "20171101", "20171108", "20171115", "20171122", "20171129", "20171206", "20171213", "20171220", "20171227", "20170105", "20170112", "20170119", "20170126", "20170202", "20170209", "20170216", "20170223", "20170302", "20170309", "20170316", "20170323", "20170330", "20170406", "20170413", "20170420", "20170427", "20170504", "20170511", "20170518", "20170525", "20170601", "20170608", "20170615", "20170622", "20170629", "20170706", "20170713", "20170720", "20170727", "20170803", "20170810", "20170817", "20170824", "20170831", "20170907", "20170914", "20170921", "20170928", "20171012", "20171019", "20171026", "20171102", "20171109", "20171116", "20171123", "20171130", "20171207", "20171214", "20171221", "20171228", "20170106", "20170113", "20170120", "20170127", "20170203", "20170210", "20170217", "20170224", "20170303", "20170310", "20170317", "20170324", "20170331", "20170407", "20170414", "20170421", "20170428", "20170505", "20170512", "20170519", "20170526", "20170602", "20170609", "20170616", "20170623", "20170630", "20170707", "20170714", "20170721", "20170728", "20170804", "20170811", "20170818", "20170825", "20170901", "20170908", "20170915", "20170922", "20170929", "20171006", "20171013", "20171020", "20171027", "20171103", "20171110", "20171117", "20171124", "20171201", "20171208", "20171215", "20171222", "20171229"]
shuffle(all_date)
test_date=[]
k_folds=5
for i in range(k_folds):
    test_date.append(all_date[i*len(all_date)//k_folds:(i+1)*len(all_date)//k_folds])
from datetime import datetime
def get_train_test(input_df,a):
    df_train = pd.DataFrame()
    
    temp=[]
    for i in range(k_folds):
        if i!=a:
            temp=temp+test_date[i]

    for i in temp:
        if datetime.strptime(i,"%Y%m%d").date() not in date_exclude:
            df_train = pd.concat([df_train,input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
        
    df_train=df_train[~df_train.index.duplicated(keep='first')]
    df_test = pd.DataFrame()
    for i in test_date[a]:
        if datetime.strptime(i,"%Y%m%d").date() not in date_exclude:
            df_test = pd.concat([df_test, input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
    df_test=df_test[~df_test.index.duplicated(keep='first')]
    return df_train, df_test

all_testing_days_lt = [testing_days1, testing_days2, testing_days3, testing_days4, testing_days5, testing_days6, testing_days7, testing_days8, testing_days9, testing_days10, testing_days11, testing_days12, testing_days13, testing_days14, testing_days15, testing_days16, testing_days17, testing_days18, testing_days19, testing_days20]
weekdays=["MON",'TUE',"WED","THU","FRI","SAT","SUN"]
#holidays=[datetime.date(2017,1,1),datetime.date(2017,1,27),datetime.date(2017,1,28),datetime.date(2017,1,29),datetime.date(2017,1,30),datetime.date(2017,1,31),datetime.date(2017,2,1),datetime.date(2017,2,28),datetime.date(2017,4,3),datetime.date(2017,4,4),datetime.date(2017,5,1),datetime.date(2017,5,30),datetime.date(2017,10,4),datetime.date(2017,10,10)]
sensors=["155","182","248","264","293","339","376","413","467","509"]
percentage_list=["50%","84%"]
start="155"
end="339"
delay_start='339'
delay_end='376'
name=start+'_'+end+'_median_latency_5'
target_columns = [name]
#name=starting point only,actually is a segment from it to next detector
#end point of past latency=509
#acc=accumulation, en=entry rate, ex=exit rate, in=in flux, out=out flux, p1=past 1 min

############generating input columns####################
start_no=sensors.index(start)
end_no=sensors.index(end)
input_columns_new=[start+"_"+end+"_p_l_median"]

for list_no in range(start_no+1,end_no+1):
    for past in percentage_list:
        name_1=start+"_"+sensors[list_no]+'_median_speed_past_'+past
        #df[name_1]=input_df[name_1]
        input_columns_new.append(name_1)
if end_no-start_no>1:
    for past in percentage_list:
        name_2=sensors[start_no+1]+"_"+end+'_median_speed_past_'+past
        input_columns_new.append(name_2)
        #df[name_2]=input_df[name_2]
#######add flux and accumulation to the input#################
bottleneck=sensors[(sensors.index(start)+sensors.index(end))//2]
for i in range(1,4):
    input_columns_new.append(start+"_out_past_"+str(i))
    if sensors.index(end)-sensors.index(start)>1:
        input_columns_new.append(bottleneck+"_out_past_"+str(i))
input_columns_new.append(start+"_"+end+"_acc")
if sensors.index(end)-sensors.index(start)>1:
    input_columns_new.append(bottleneck+"_"+sensors[sensors.index(bottleneck)+1]+"_acc")
##########excluding public holidays#################################
"""
for holi in holidays:
    df_wd=df_wd[df_wd.index.date!=holi]
    """
#df=df[df.index!=datetime.date(2017,10,19)]
################input for time delay####################################################
delay_start_no=sensors.index(delay_start)
delay_end_no=sensors.index(delay_end)
#############add p_l and speed################
delay_input=[delay_start+"_"+delay_end+"_p_l_median"]
###########p_l_median_x_min_later##########
"""
if delay_minute!=0:
    delay_input.append(delay_start+"_"+delay_end+"_p_l_median_"+str(delay_minute)+"_min_later")
"""
##################################################
for list_no in range(delay_start_no+1,delay_end_no+1):
    for past in percentage_list:
        name_1=delay_start+"_"+sensors[list_no]+'_median_speed_past_'+past
        #df[name_1]=input_df[name_1]
        delay_input.append(name_1)
if delay_end_no-delay_start_no>1:
    for past in percentage_list:
        name_2=sensors[delay_start_no+1]+"_"+delay_end+'_median_speed_past_'+past
        delay_input.append(name_2)
        #df[name_2]=input_df[name_2]
#######add flux and accumulation#################
bottleneck=sensors[(sensors.index(delay_start)+sensors.index(delay_end))//2]
for i in range(1,4):
    delay_input.append(delay_start+"_out_past_"+str(i))
    if sensors.index(delay_end)-sensors.index(delay_start)>1:
        delay_input.append(bottleneck+"_out_past_"+str(i))
delay_input.append(delay_start+"_"+delay_end+"_acc")
if sensors.index(delay_end)-sensors.index(delay_start)>1:
    delay_input.append(bottleneck+"_"+sensors[sensors.index(bottleneck)+1]+"_acc")
whole_tar=[start+"_"+delay_end+"_median_latency_5"]
def avg(li):
        return sum(li)/len(li)
 
"""
def separate(row):
        if row['155_182_acc']>140:
            return 1
        else:
            return 0
n_cluster=2
save=pd.DataFrame()
"""
for glhf in range(15):
    delay_minute=glhf
    ############target columns#######################
    if delay_minute!=0:
        tar=[delay_start+"_"+delay_end+"_median_latency_5_"+str(delay_minute)+"_min_later"]
    else:
        tar=[delay_start+"_"+delay_end+"_median_latency_5"]
    ##reading pickle#####################
    out_acc=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_out_acc.pickle")
    pl=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_p_l.pickle")
    speed=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_speed.pickle")
    df_wd=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_latency.pickle")
    """
    if delay_minute!=0:
        df_wd=df_wd.rename(columns={delay_start+'_'+delay_end+'_median_latency_5':delay_start+'_'+delay_end+'_median_latency_5_'+})
        """
    df_wd=pd.concat([df_wd,out_acc,pl,speed],axis=1)
    df_wd=df_wd[list(set(input_columns_new+target_columns+delay_input+tar+whole_tar))]
    df_wd=df_wd.reset_index()
    df_wd['index']=pd.to_datetime(df_wd['index'],utc=True)
    df_wd.set_index('index',inplace=True)
    ##################start prediction#######################################################
    MSE_test_list=[[],[],[],[],[]]
    MAPE_test_list=[[],[],[],[],[]]
    first_seg_MSE_list=[[],[],[],[],[]]
    second_seg_MSE_list=[[],[],[],[],[]]

    for a in range(5):
        df_train_or, df_test_or = get_train_test(df_wd, a)
        df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
        df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
        df_train = df_train_or.dropna()
        df_test = df_test_or.dropna()
        #df_train['GMM']=df_train.apply(lambda row: separate(row),axis=1)
        #df_test['GMM']=df_test.apply(lambda row: separate(row),axis=1)

        df_train_pred  = pd.DataFrame()
        df_test_pred   = pd.DataFrame()
        XGB()
        #GMM(2)
        df_train_pred = df_train_pred.sort_index()
        df_test_pred = df_test_pred.sort_index()
        df_train=df_train.sort_index()
        df_test=df_test.sort_index()
        first_seg_error=mean_squared_error(df_test[target_columns],df_test_pred['pred_new'])
        first_seg_MSE_list[0].append(first_seg_error)
        
        predict_delay()
        df_train_pred = df_train_pred.sort_index()
        df_test_pred = df_test_pred.sort_index()
        df_train=df_train.sort_index()
        df_test=df_test.sort_index()
        second_seg_error=mean_squared_error(df_test[tar],df_test_pred['delay_pred_new'])
        second_seg_MSE_list[0].append(second_seg_error)
        
        
        df_train['pred_new'] = df_train_pred['pred_new']+df_train_pred['delay_pred_new']
        df_test['pred_new']  = df_test_pred['pred_new']+df_test_pred['delay_pred_new']
        #save=pd.concat([save,df_test['pred_new']],axis=1)   
        
        MSE_train = mean_squared_error(df_train[whole_tar], df_train['pred_new'])
        #MAPE_train = MAPE(df_train[whole_tar], df_train['pred_new'])
        MSE_test = mean_squared_error(df_test[whole_tar], df_test['pred_new'])
        #MAPE_test = MAPE(df_test[whole_tar], df_test['pred_new'])
        
        MSE_test_list[0].append(MSE_test)
        #MAPE_test_list[w].append(MAPE_test)
        #print(w,a,MSE_train,MAPE_train,"%",MSE_test,MAPE_test,"%")
        #print(a,MSE_train,MSE_test)
        """
        for month in range(1,13,1):
            temp = df_test.loc[df_test.index.month == month]
            prev_day=0
            for day in temp.index.day:
                if(prev_day==day):
                    continue
                case = temp.loc[temp.index.day ==day]
                case_1=pd.DataFrame()
                case_1[name] = case[whole_tar] 
                case_1['XGB'] = case['pred_new'] 
                
                case_1_x=[]
                for i in case_1.index.time:
                    case_1_x.append(i.hour*60+i.minute)
                plt.plot(case_1_x,case_1[name]);
                plt.plot(case_1_x,case_1['XGB']);
                plt.ylabel('Latency / second');
                plt.legend((name,'predict latency'), loc='upper left');
                plt.title(str(day)+'/'+str(month)+'/2017 ('+weekdays[datetime.datetime.weekday(datetime.datetime(2017,month,day))]+")_("+str(delay_minute)+"_min_delay"+')');
                plt.savefig("C:/Users/CHEN/Desktop/155_182_248_plot/"+'2017'+'-'+str(month)+'-'+str(day)+'_'+start+"_"+end+"_"+delay_end+"_"+str(delay_minute)+"_min_delay")
                plt.show();
                prev_day=day
                """
    print("delay min: {}, MSE: {}, 2nd seg MSE: {}".format(delay_minute,avg(MSE_test_list[0]),avg(second_seg_MSE_list[0])))

"""
for w in range(5):
    df_wd = sep_weekday(df, [w])
    df_wd=df_wd[list(set(input_columns_new+target_columns+delay_input+tar+whole_tar))]
    for a in range(20):
        df_train_or, df_test_or = sep_train_test(df_wd, weekday=[w], testing_days=all_testing_days_lt[a])
        df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
        df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
        df_train = df_train_or.dropna()
        df_test = df_test_or.dropna()
        #df_train['GMM']=df_train.apply(lambda row: separate(row),axis=1)
        #df_test['GMM']=df_test.apply(lambda row: separate(row),axis=1)

        df_train_pred  = pd.DataFrame()
        df_test_pred   = pd.DataFrame()
        XGB()
        #GMM(2)
        df_train_pred = df_train_pred.sort_index()
        df_test_pred = df_test_pred.sort_index()
        first_seg_error=mean_squared_error(df_test[target_columns],df_test_pred['pred_new'])
        first_seg_MSE_list[w].append(first_seg_error)
        
        predict_delay()
        second_seg_error=mean_squared_error(df_test[tar],df_test_pred['delay_pred_new'])
        second_seg_MSE_list[w].append(second_seg_error)
        
        
        df_train['pred_new'] = df_train_pred['pred_new']+df_train_pred['delay_pred_new']
        df_test['pred_new']  = df_test_pred['pred_new']+df_test_pred['delay_pred_new']
        #save=pd.concat([save,df_test['pred_new']],axis=1)   
        
        MSE_train = mean_squared_error(df_train[whole_tar], df_train['pred_new'])
        #MAPE_train = MAPE(df_train[whole_tar], df_train['pred_new'])
        MSE_test = mean_squared_error(df_test[whole_tar], df_test['pred_new'])
        #MAPE_test = MAPE(df_test[whole_tar], df_test['pred_new'])
        
        MSE_test_list[w].append(MSE_test)
        #MAPE_test_list[w].append(MAPE_test)
        #print(w,a,MSE_train,MAPE_train,"%",MSE_test,MAPE_test,"%")
        print(w,a,MSE_train,MSE_test)

        for month in range(1,13,1):
            temp = df_test.loc[df_test.index.month == month]
            prev_day=0
            for day in temp.index.day:
                if(prev_day==day):
                    continue
                case = temp.loc[temp.index.day ==day]
                case_1=pd.DataFrame()
                case_1[name] = case[whole_tar] 
                case_1['XGB'] = case['pred_new'] 
                
                case_1_x=[]
                for i in case_1.index.time:
                    case_1_x.append(i.hour*60+i.minute)
                plt.plot(case_1_x,case_1[name]);
                plt.plot(case_1_x,case_1['XGB']);
                plt.ylabel('Latency / second');
                plt.legend((name,'predict latency'), loc='upper left');
                plt.title(str(day)+'/'+str(month)+'/2017 ('+weekdays[datetime.datetime.weekday(datetime.datetime(2017,month,day))]+")_("+str(delay_minute)+"_min_delay"+')');
                plt.savefig("C:/Users/CHEN/Desktop/155_182_248_plot/"+'2017'+'-'+str(month)+'-'+str(day)+'_'+start+"_"+end+"_"+delay_end+"_"+str(delay_minute)+"_min_delay")
                plt.show();
                prev_day=day
for w in range(5):
    print("weekday: {},MSE: {}, MAPE: {}".format(w,avg(MSE_test_list[w]),avg(MAPE_test_list[w])))
    print("weekday: {},whole_MSE: {}, first seg MSE: {}, second seg MSE: {}".format(w,avg(MSE_test_list[w]),avg(first_seg_MSE_list[w]),avg(second_seg_MSE_list[w])))
    """