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
import random

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
    X_test_new  = df_test[input_columns_new]

    y_train_new = df_train[target_columns]
    y_test_new = df_test[target_columns]

    pred_model = xgb.XGBRegressor()
    pred_model.fit(X_train_new, y_train_new)
    y_train_pred = pred_model.predict(X_train_new)
    X_train_new['pred_new'] = y_train_pred
    y_test_pred = pred_model.predict(X_test_new)
    X_test_new['pred_new'] = y_test_pred
    
    df_train_pred  =   X_train_new
    df_test_pred   =   X_test_new
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
start="155"
end="509"
name=start+'_'+end+'_median_latency_5'
target_columns = [name]
weekdays=["MON",'TUE',"WED","THU","FRI","SAT","SUN"]
holidays=[datetime.date(2017,1,27),datetime.date(2017,1,28),datetime.date(2017,1,29),datetime.date(2017,1,30),datetime.date(2017,1,31),datetime.date(2017,2,1),datetime.date(2017,2,28),datetime.date(2017,4,3),datetime.date(2017,4,4),datetime.date(2017,5,1),datetime.date(2017,5,30),datetime.date(2017,10,4),datetime.date(2017,10,10)]
#name=starting point only,actually is a segment from it to next detector
#end point of past latency=509
#acc=accumulation, en=entry rate, ex=exit rate, in=in flux, out=out flux, p1=past 1 min
#date_exclude=[datetime.date(2017,7,28),datetime.date(2017,5,25),datetime.date(2017,8,10),datetime.date(2017,8,30),datetime.date(2017,8,11),datetime.date(2017,4,27),datetime.date(2017,7,6),datetime.date(2017,8,10),datetime.date(2017,3,28),datetime.date(2017,6,22),datetime.date(2017,4,26),datetime.date(2017,12,21),datetime.date(2017,7,13),datetime.date(2017,4,26),datetime.date(2017,7,5),datetime.date(2017,9,29),datetime.date(2017,12,21),datetime.date(2017,3,31),datetime.date(2017,7,7),datetime.date(2017,9,21),datetime.date(2017,9,22),datetime.date(2017,1,13),datetime.date(2017,8,14),datetime.date(2017,6,14),datetime.date(2017,5,17),datetime.date(2017,1,4),datetime.date(2017,9,22),datetime.date(2017,6,2),datetime.date(2017,2,21),datetime.date(2017,1,4),datetime.date(2017,1,11),datetime.date(2017,1,25), datetime.date(2017,2,6), datetime.date(2017,5,17), datetime.date(2017,9,27),datetime.date(2017,1,19), datetime.date(2017,10,19),datetime.date(2017,1,13), datetime.date(2017,1,26)]
#date_exclude=[datetime.date(2017,9,1),datetime.date(2017,8,10),datetime.date(2017,7,28),datetime.date(2017,8,25),datetime.date(2017,6,27)]
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
            df_train = pd.concat([df_train,input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
        
    df_train=df_train[~df_train.index.duplicated(keep='first')]
    df_test = pd.DataFrame()
    for i in test_date[a]:
        #if datetime.strptime(i,"%Y%m%d").date() not in date_exclude:
            df_test = pd.concat([df_test, input_df[input_df.index.date==datetime.strptime(i,"%Y%m%d").date()]])
    df_test=df_test[~df_test.index.duplicated(keep='first')]
    return df_train, df_test
#########generating input columns#########
sensors=["155","182","248","264","293","339","376","413","467","509"]
percentage_list=["50%","84%"]
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

##reading pickle#####################
out_acc=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_out_acc.pickle")
pl=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_p_l.pickle")
speed=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_speed.pickle")
#df_wd=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_latency.pickle")
df_wd=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/new_output_df_155-509.pickle")
df_wd=pd.concat([df_wd,out_acc,pl,speed],axis=1)
df_wd=df_wd[list(set(input_columns_new+target_columns))]
df_wd=df_wd.reset_index()
df_wd['index']=pd.to_datetime(df_wd['index'],utc=True)
df_wd.set_index('index',inplace=True)
################start predicting####################################################
MSE_test_list=[[],[],[],[],[]]
MAPE_test_list=[[],[],[],[],[]]
def avg(li):
    return sum(li)/len(li)
def separate(row):
    if row['155_182_acc']>140:
        return 1
    else:
        return 0
n_cluster=2
save=pd.DataFrame()

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
        
        df_train['pred_new'] = df_train_pred['pred_new']
        df_test['pred_new']  = df_test_pred['pred_new']
        save=pd.concat([save,df_test['pred_new']],axis=1)   

        MSE_train = mean_squared_error(df_train[name], df_train['pred_new'])
        MAPE_train = MAPE(df_train[name], df_train['pred_new'])
        MSE_test = mean_squared_error(df_test[name], df_test['pred_new'])
        MAPE_test = MAPE(df_test[name], df_test['pred_new'])
        
        MSE_test_list[0].append(MSE_test)
        MAPE_test_list[0].append(MAPE_test)
        print(a,MSE_train,MAPE_train,"%",MSE_test,MAPE_test,"%")
"""
        for month in range(1,13,1):
            temp = df_test.loc[df_test.index.month == month]
            prev_day=0
            for day in temp.index.day:
                if(prev_day==day):
                    continue
                case = temp.loc[temp.index.day ==day]
                case_1=pd.DataFrame()
                case_1[name] = case[name] 
                case_1['XGB'] = case['pred_new'] 
                
                case_1_x=[]
                for i in case_1.index.time:
                    case_1_x.append(i.hour*60+i.minute)
                plt.plot(case_1_x,case_1[name]);
                plt.plot(case_1_x,case_1['XGB']);
                plt.ylabel('Latency / second');
                plt.legend((name,'predict latency'), loc='upper left');
                plt.title(str(day)+'/'+str(month)+'/2017 ('+weekdays[datetime.weekday(datetime(2017,month,day))]+')');
                plt.savefig("C:/Users/CHEN/Desktop/155_509/"+'2017'+'-'+str(month)+'-'+str(day)+'_'+start+"_"+end)
                plt.show();
                prev_day=day
                """
print("MSE: {}, MAPE: {}".format(avg(MSE_test_list[0]),avg(MAPE_test_list[0])))