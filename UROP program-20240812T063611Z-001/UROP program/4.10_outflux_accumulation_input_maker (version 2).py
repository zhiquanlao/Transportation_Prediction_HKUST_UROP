import pandas as pd
import numpy as np
import time
import datetime as dt
from zipfile import ZipFile
import os
start_time=time.time()
weekdays=["MON","TUE","WED","THU","FRI"]
#list of sensors where data is calculated: 
#if you want to calculate data between 155 and 182 for example, be sure to include all sensors between 155 and 182 (inclusive) for the code to work!
#both lists have to be changed!
#sensors_list=["01F0155S","01F0182S","01F0248S","01F0264S","01F0293S","01F0339S","01F0376S","01F0413S","01F0467S","01F0509S"]
#sensors_short=["155","182","248","264","293","339","376","413","467","509"]
sensors_list=["01F0155S","01F0182S","01F0248S","01F0264S","01F0293S","01F0339S"]
sensors_short=["155","182","248","264","293","339"]
#pqr=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_out_acc.pickle")   
#pqr=pqr.filter(like='acc',axis=1) 
for w in weekdays:  #set w to integer if only one weekday data is needed
    all_days=pd.date_range("20170101","20171231",freq="W-"+w)   #generate range of dates for a weekday
    all_days=all_days.strftime('%Y%m%d')    #change the variable type of dates generated
    for date in all_days:   #get data for each day
        file="C:/Users/CHEN/Desktop/latency/"+date+"_latency.pickle" #this dataframe can be downloaded in onedrive in "latency.zip"
        #be careful not to mix up the latency file that you have generated with the one above!
        
        with ZipFile("C:/Users/CHEN/Desktop/latency.zip","r") as zipObj:
            fileName=date+"_latency.pickle"
            zipObj.extract(fileName,"C:/Users/CHEN/Desktop/latency/")
            
        daily_df=pd.read_pickle(file)
        whole_day=pd.date_range(date+'054500',date+'230000',freq="T")   #selecting section of time needed to be calculated
        output_df=pd.DataFrame({"time":whole_day})#generated empty dataframe
        
        for start_no in range(len(sensors_short)-2,len(sensors_short)-1):  #for every sensor in the list
            st_sensor=sensors_list[start_no]
            ed_sensor=sensors_list[start_no+1]
            for t in whole_day:  #for each minute
                name2=sensors_short[start_no]+"_out" #calculation of outflux
                start_t=t
                end_t=start_t+dt.timedelta(minutes=1)
                out_target=daily_df.loc[(daily_df[ed_sensor]>=start_t)&(daily_df[ed_sensor]<end_t)] #selecting vehicles passing thorugh ending sensor between the designated time 
                single_target=len(out_target.loc[(out_target[st_sensor].isnull())]) #remove vehicles that are not through traffic
                output_df.loc[output_df["time"]==t,[name2]]=(len(out_target)-single_target)
                
                name2_1=sensors_short[start_no]+"_in" #calculation of influx
                out_target=daily_df.loc[(daily_df[st_sensor]>=start_t)&(daily_df[st_sensor]<end_t)] #selecting vehicles passing thorugh starting sensor between the designated time 
                single_target=len(out_target.loc[(out_target[ed_sensor].isnull())]) #remove vehicles that are not through traffic
                output_df.loc[output_df["time"]==t,[name2_1]]=(len(out_target)-single_target)
                
                
        for start_no in range(len(sensors_short)-2,len(sensors_short)-1):
            name2o=sensors_short[start_no]+"_out"
            for past in range(1,4,1):   #create past time data 
                name2p=sensors_short[start_no]+"_out_past_"+str(past)
                output_df[name2p]=output_df[name2o].shift(past).copy()
                
        ###########change#################################
        #output_df=pd.concat([output_df,pqr[date]],axis=1)
        ##################################################
        """
        start_day=True
        for t in whole_day:  #calulate accumulation of each minute
            end_t=t+dt.timedelta(minutes=1)
            for start_no_1 in range(len(sensors_short)-1):
                end_no=start_no_1+1
                st_sensor=sensors_list[start_no_1]
                ed_sensor=sensors_list[end_no]
                name3=sensors_short[start_no_1]+"_"+sensors_short[end_no]+"_acc"
                name3f_up=sensors_short[start_no_1]+"_in"
                name3f_down=sensors_short[start_no_1]+"_out"
                if start_day==True:
                    acc_prev=len(daily_df.loc[(daily_df[st_sensor]<=t)&(daily_df[ed_sensor]>=t)]) #approximate number of vehicles in segment at the start of day
                else:
                    acc_prev=output_df.loc[output_df["time"]==t-dt.timedelta(minutes=1),[name3]].values #accumulation of vehicles in the previous minute
                acc=output_df.loc[output_df["time"]==t,[name3f_up]].values-output_df.loc[output_df["time"]==t,[name3f_down]].values+acc_prev #influx+outflux=change of accumulation
                if acc<0:
                    acc=0
                output_df.loc[output_df["time"]==t,[name3]]=acc
            start_day=False
         """
        start_day=True
        for t in whole_day:  #calulate accumulation of each minute
            end_t=t+dt.timedelta(minutes=1)
            for start_no_1 in range(len(sensors_short)-2,len(sensors_short)-1):
                end_no=start_no_1+1
                st_sensor=sensors_list[start_no_1]
                ed_sensor=sensors_list[end_no]
                name3=sensors_short[start_no_1]+"_"+sensors_short[end_no]+"_acc"
                name3f_up=sensors_short[start_no_1]+"_in"
                name3f_down=sensors_short[start_no_1]+"_out"
                if start_day==True:
                    acc_prev=len(daily_df.loc[(daily_df[st_sensor]<=t)&(daily_df[ed_sensor]>=t)]) #approximate number of vehicles in segment at the start of day
                else:
                    acc_prev=output_df.loc[output_df["time"]==t-dt.timedelta(minutes=1),[name3]].values #accumulation of vehicles in the previous minute
                acc=output_df.loc[output_df["time"]==t,[name3f_up]].values-output_df.loc[output_df["time"]==t,[name3f_down]].values+acc_prev #influx+outflux=change of accumulation
                if acc<0:
                    acc=0
                output_df.loc[output_df["time"]==t,[name3]]=acc
            start_day=False
        """
        for start_no_2 in range(len(sensors_short)-2):  #calculate accumulation of longer segments by adding up sub-single segments
            for end_no_2 in range(len(sensors_short)-1,len(sensors_short)):
                name4=sensors_short[start_no_2]+"_"+sensors_short[end_no_2]+"_acc"
                for start_sub in range(start_no_2,end_no_2):
                    sub_name=sensors_short[start_sub]+"_"+sensors_short[start_sub+1]+"_acc"
                    if start_sub==start_no_2:
                        acc_sum=output_df[sub_name]
                    else:
                        acc_sum=acc_sum+output_df[sub_name]
                output_df[name4]=acc_sum
               """
        print(w,date,int(time.time()-start_time))
        
        output_df=output_df.loc[15:]
        output_df=output_df.set_index("time")
        print(date,"Done!")
        output_df.to_pickle("C:/Users/CHEN/Desktop/abc/"+date+"_out_acc_inputs_ver.2.pickle")
        
        os.remove(file)
        
end_time=time.time()
duration=end_time-start_time
duration_hour=duration//3600
duration_min=(duration%3600)//60
duration_sec=int((duration%3600)%60)
print(duration_hour,"hour",duration_min,"min",duration_sec,"sec")
"""
for date in all_days:   #get data for each day
    file="C:/Users/CHEN/Desktop/latency/"+date+"_latency.pickle" #this dataframe can be downloaded in onedrive in "latency.zip"
    #be careful not to mix up the latency file that you have generated with the one above!
    daily_df=pd.read_pickle(file)
    whole_day=pd.date_range(date+'054500',date+'230000',freq="T")   #selecting section of time needed to be calculated
    output_df=pd.DataFrame({"time":whole_day})#generated empty dataframe
    
    for start_no in range(len(sensors_short)-1):  #for every sensor in the list
        st_sensor=sensors_list[start_no]
        ed_sensor=sensors_list[start_no+1]
        for t in whole_day:  #for each minute
            name2=sensors_short[start_no]+"_out" #calculation of outflux
            start_t=t
            end_t=start_t+dt.timedelta(minutes=1)
            out_target=daily_df.loc[(daily_df[ed_sensor]>=start_t)&(daily_df[ed_sensor]<end_t)] #selecting vehicles passing thorugh ending sensor between the designated time 
            single_target=len(out_target.loc[(out_target[st_sensor].isnull())]) #remove vehicles that are not through traffic
            output_df.loc[output_df["time"]==t,[name2]]=(len(out_target)-single_target)
            
            name2_1=sensors_short[start_no]+"_in" #calculation of influx
            out_target=daily_df.loc[(daily_df[st_sensor]>=start_t)&(daily_df[st_sensor]<end_t)] #selecting vehicles passing thorugh starting sensor between the designated time 
            single_target=len(out_target.loc[(out_target[ed_sensor].isnull())]) #remove vehicles that are not through traffic
            output_df.loc[output_df["time"]==t,[name2_1]]=(len(out_target)-single_target)
            
            
    for start_no in range(len(sensors_short)-1):
        name2o=sensors_short[start_no]+"_out"
        for past in range(1,4,1):   #create past time data 
            name2p=sensors_short[start_no]+"_out_past_"+str(past)
            output_df[name2p]=output_df[name2o].shift(past).copy()
            
    start_day=True
    for t in whole_day:  #calulate accumulation of each minute
        end_t=t+dt.timedelta(minutes=1)
        for start_no_1 in range(len(sensors_short)-1):
            end_no=start_no_1+1
            st_sensor=sensors_list[start_no_1]
            ed_sensor=sensors_list[end_no]
            name3=sensors_short[start_no_1]+"_"+sensors_short[end_no]+"_acc"
            name3f_up=sensors_short[start_no_1]+"_in"
            name3f_down=sensors_short[start_no_1]+"_out"
            if start_day==True:
                acc_prev=len(daily_df.loc[(daily_df[st_sensor]<=t)&(daily_df[ed_sensor]>=t)]) #approximate number of vehicles in segment at the start of day
            else:
                acc_prev=output_df.loc[output_df["time"]==t-dt.timedelta(minutes=1),[name3]].values #accumulation of vehicles in the previous minute
            acc=output_df.loc[output_df["time"]==t,[name3f_up]].values-output_df.loc[output_df["time"]==t,[name3f_down]].values+acc_prev #influx+outflux=change of accumulation
            if acc<0:
                acc=0
            output_df.loc[output_df["time"]==t,[name3]]=acc
        start_day=False
        
    for start_no_2 in range(len(sensors_short)-2):  #calculate accumulation of longer segments by adding up sub-single segments
        for end_no_2 in range(start_no_2+2,len(sensors_short)):
            name4=sensors_short[start_no_2]+"_"+sensors_short[end_no_2]+"_acc"
            for start_sub in range(start_no_2,end_no_2):
                sub_name=sensors_short[start_sub]+"_"+sensors_short[start_sub+1]+"_acc"
                if start_sub==start_no_2:
                    acc_sum=output_df[sub_name]
                else:
                    acc_sum=acc_sum+output_df[sub_name]
            output_df[name4]=acc_sum
            
    print(w,date,int(time.time()-start_time))
    
    output_df=output_df.loc[15:]
    output_df=output_df.set_index("time")
    print(date,"Done!")
    output_df.to_pickle("C:/Users/CHEN/Desktop/abc/"+date+"_out_acc_inputs_ver.2.pickle")
"""