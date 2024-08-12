# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:13:12 2022

@author: CHEN
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import mixture

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import datetime
import time
import csv
weekdays=["MON",'TUE',"WED","THU","FRI","SAT","SUN"]

division=["155","182","248"]#e.g.155_182_248, input'155', '182', '248'
no_division=len(division)-1
delay_list=[0,0,0,0,0,0,0,0,0]#e.g.155_182_0min,182_248_3min, input 0,3
whole_start=division[0]
whole_end=division[-1]
whole_tar=[whole_start+'_'+whole_end+'_median_latency_5']
all_targets=[]
for i in range(no_division):
    if delay_list[i]==0:
        stri=division[i]+'_'+division[i+1]+'_median_latency_5'
    else:
        stri=division[i]+'_'+division[i+1]+'_median_latency_5_'+str(delay_list[i])+'_min_later'
    all_targets.append(stri)
sensors=["155","182","248","264","293","339","376","413","467","509"]
percentage_list=["50%","84%"]
segment_MSE_list=[]
segment_MAPE_list=[]
whole_MSE_list=[]
whole_MAPE_list=[]
for i in range(no_division):
    segment_MSE_list.append([])
    segment_MAPE_list.append([])
def avg(li):
        return sum(li)/len(li)
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
def XGB():
    global df_train_pred
    global df_test_pred

    X_train_new = df_train[input_columns_new]
    X_test_new  = df_test[input_columns_new]

    y_train_new = df_train[target_columns]
    y_test_new = df_test[target_columns]
    stri=target_columns[0]
    pred_model = xgb.XGBRegressor()
    
    
    #pred_model.fit(X_train_new, y_train_new)
    pred_model.fit(np.array(X_train_new), np.array(y_train_new))
    #y_train_pred = pred_model.predict(X_train_new)
    y_train_pred = pred_model.predict(np.array(X_train_new))
    X_train_new[stri+'_pred'] = y_train_pred
    #y_test_pred = pred_model.predict(X_test_new)
    y_test_pred = pred_model.predict(np.array(X_test_new))
    
    
    X_test_new[stri+'_pred'] = y_test_pred
    
    df_train_pred[stri+'_pred']  =   X_train_new[stri+'_pred']
    df_test_pred[stri+'_pred']   =   X_test_new[stri+'_pred']
######################standard input##################

def generate_in_out(start,end,delay_minute):
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
    if delay_minute!=0:
        target_columns=[start+'_'+end+"_median_latency_5_"+str(delay_minute)+"_min_later"]
    else:
        target_columns=[start+'_'+end+"_median_latency_5"]
    return input_columns_new,target_columns
"""
##################baiyue's input###############################
def generate_in_out(start,end,delay_minute):
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
    input_columns_new.append(start+'_in')
    input_columns_new.append(start+"_"+end+"_acc")
    if sensors.index(end)-sensors.index(start)>1:
        input_columns_new.append(bottleneck+"_"+sensors[sensors.index(bottleneck)+1]+"_acc")
    if delay_minute!=0:
        target_columns=[start+'_'+end+"_median_latency_5_"+str(delay_minute)+"_min_later"]
    else:
        target_columns=[start+'_'+end+"_median_latency_5"]
    return input_columns_new,target_columns
"""
#########train test set, to be changed##################
"""
date_range = [d for d in pd.date_range(start="2017-01-01", end="2017-12-31") if d.weekday() in [0,1,2,3,4]]

date_range=[day.strftime('%Y%m%d') for day in date_range]

random.seed(1)

random.shuffle(date_range)

test_df=pd.DataFrame()

date_range = [date_range[i::5] for i in range(5)]
"""
#date_exclude=[datetime.date(2017,12,29),datetime.date(2017,6,29),datetime.date(2017,1,13),datetime.date(2017,1,4),datetime.date(2017,3,10),datetime.date(2017,12,22),datetime.date(2017,7,28),datetime.date(2017,8,25),datetime.date(2017,5,17),datetime.date(2017,6,23),datetime.date(2017,6,27),datetime.date(2017,8,25),datetime.date(2017,8,10),datetime.date(2017,9,1),datetime.date(2017,7,28),datetime.date(2017,5,25),datetime.date(2017,8,10),datetime.date(2017,8,30),datetime.date(2017,8,11),datetime.date(2017,4,27),datetime.date(2017,7,6),datetime.date(2017,8,10),datetime.date(2017,3,28),datetime.date(2017,6,22),datetime.date(2017,4,26),datetime.date(2017,12,21),datetime.date(2017,7,13),datetime.date(2017,4,26),datetime.date(2017,7,5),datetime.date(2017,9,29),datetime.date(2017,12,21),datetime.date(2017,3,31),datetime.date(2017,7,7),datetime.date(2017,9,21),datetime.date(2017,9,22),datetime.date(2017,1,13),datetime.date(2017,8,14),datetime.date(2017,6,14),datetime.date(2017,5,17),datetime.date(2017,1,4),datetime.date(2017,9,22),datetime.date(2017,6,2)]
all_date=[d for d in pd.date_range(start="2017-01-01", end="2017-12-31") if d.weekday() in [0,1,2,3,4]]
all_date=[day.strftime('%Y%m%d') for day in all_date]
random.seed(1)
random.shuffle(all_date)
k_folds=5
all_date=[all_date[i::k_folds] for i in range(k_folds)]
test_date=[]
for i in range(k_folds):
    test_date.append(all_date[i])
from datetime import datetime
def get_train_test(input_df,a):
    df_train = pd.DataFrame()
    
    temp=[]
    for i in range(k_folds):
        if i!=a:
            temp=temp+test_date[i]

    for i in temp:
        #if datetime.strptime(i,"%Y%m%d").date() not in date_exclude:
            try:
                df_train = pd.concat([df_train,input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
            except KeyError:
                continue
    df_train=df_train[~df_train.index.duplicated(keep='first')]
    df_test = pd.DataFrame()
    for i in test_date[a]:
        #if datetime.strptime(i,"%Y%m%d").date() not in date_exclude:
            try:
                df_test = pd.concat([df_test, input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
            except KeyError:
                continue
    df_test=df_test[~df_test.index.duplicated(keep='first')]
    return df_train, df_test
############################read files, to be changed to generate time delay#####################
input_columns_new=[]
target_columns=[]
for i in range(no_division):
     seg_start,seg_end,seg_delay=division[i],division[i+1],delay_list[i]
     temp1,temp2=generate_in_out(start=seg_start, end=seg_end, delay_minute=seg_delay)
     input_columns_new=list(set(input_columns_new+temp1))
     target_columns=list(set(target_columns+temp2))
target_columns.append(whole_start+'_'+whole_end+'_median_latency_5')
out_acc=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_out_acc.pickle")
pl=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_p_l.pickle")
speed=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_speed.pickle")
#df=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_latency.pickle")
df=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/new_output_df_155-509.pickle")
df=pd.concat([out_acc,pl,speed,df],axis=1)
df=df[list(set(input_columns_new+target_columns))]
df=df.reset_index()
df['index']=pd.to_datetime(df['index'],utc=True)
df.set_index('index',inplace=True)

"""
df=df.replace([np.inf, -np.inf], np.nan)
df=df.dropna()
"""

date_compare='20170614'
save=pd.DataFrame()


###################prediction##########################
for a in range(5):
    df_train_pred  = pd.DataFrame()
    df_test_pred   = pd.DataFrame()
    for i in range(no_division):
        seg_start,seg_end,seg_delay=division[i],division[i+1],delay_list[i]
        input_columns_new,target_columns=generate_in_out(start=seg_start, end=seg_end, delay_minute=seg_delay)
        df_wd=df[list(set(input_columns_new+target_columns+whole_tar))]
        df_train, df_test = get_train_test(df_wd, a)
        #df_train=df_train.replace([np.inf, -np.inf], np.nan)
        #df_test=df_test.replace([np.inf, -np.inf], np.nan)
        #df_train=df_train.dropna()
        #df_test=df_test.dropna()
        stri=target_columns[0]
        
        #print(len(df_train)+len(df_test))
        XGB()
        
        df_train_pred = df_train_pred.sort_index()
        df_test_pred = df_test_pred.sort_index()
        df_train=df_train.sort_index()
        df_test=df_test.sort_index()
        MSEerror=mean_squared_error(df_test[target_columns],df_test_pred[stri+'_pred'])
        
        
       # if len(df_test[df_test.index.date==datetime.strptime(date_compare,"%Y%m%d").date()])!=0:
            #save=pd.concat([save,df_test[df_test.index.date==datetime.strptime(date_compare,"%Y%m%d").date()][target_columns],df_test_pred[df_test_pred.index.date==datetime.strptime(date_compare,"%Y%m%d").date()][stri+'_pred']],axis=1)
        
        
        #MAPEerror=MAPE(df_test[target_columns],df_test_pred[stri+'_pred'])
        segment_MSE_list[i].append(MSEerror)
        #segment_MAPE_list[i].append(MAPEerror)
    
    df_test_pred['whole_pred']=0
    for col in df_test_pred.columns:
        if col!='whole_pred':
            df_test_pred['whole_pred']=df_test_pred['whole_pred']+df_test_pred[col]
    wholeMSE=mean_squared_error(df_test[whole_tar],df_test_pred['whole_pred'])
    if len(df_test[df_test.index.date==datetime.strptime(date_compare,"%Y%m%d").date()])!=0:
        save=pd.concat([save,df_test[df_test.index.date==datetime.strptime(date_compare,"%Y%m%d").date()][whole_tar],df_test_pred[df_test_pred.index.date==datetime.strptime(date_compare,"%Y%m%d").date()]['whole_pred']],axis=1)
    #wholeMAPE=MAPE(df_test[whole_tar],df_test_pred['whole_pred'])
    #print("a:{}, MSE:{}, MAPE:{}%".format(a,wholeMSE,wholeMAPE))
    print("a:{}, MSE:{}".format(a,wholeMSE))
    whole_MSE_list.append(wholeMSE)
    #whole_MAPE_list.append(wholeMAPE)
for i in range(no_division):
    print(all_targets[i],end=": ")
    #print("MSE:{}, MAPE:{}%".format(avg(segment_MSE_list[i]),segment_MAPE_list[i]))
    print("MSE:{}".format(avg(segment_MSE_list[i])))
    #print("whole: MSE:{}, MAPE:{}%".format(avg(whole_MSE_list),avg(whole_MAPE_list)))
print("whole: MSE:{}".format(avg(whole_MSE_list)))


#save.to_pickle("C:/Users/CHEN/Desktop/"+date_compare+"_multi_prediction.pickle")