import pandas as pd
import numpy as np
import time
import datetime as dt
from zipfile import ZipFile
import os
start_time=time.time()
weekdays=["MON","TUE","WED","THU","FRI"] #change content of this list to alter the weekday of data to be generated
#sensors_short=["155","182","248","264","293","339","376","413","467","509"]
sensors_short=["155","182","248","264","293","376"]# change this list to alter the segment of data (remember to include all sensors in between the segment!)
#e.g. for 339-413, the list should be ["339","376","413"]
percentage_list=["16%","50%","84%"]#percentile of time interval

for w in weekdays:#for each weekday in the list
    all_days=pd.date_range("20170101","20171231",freq="W-"+w)#change this line to alter the date range 
    all_days=all_days.strftime('%Y%m%d')
    for date in all_days: #for each day
        file="C:/Users/CHEN/Desktop/latency/"+date+"_latency.pickle" #this latency file is the same as that of the outflux and accumulation code
        with ZipFile("C:/Users/CHEN/Desktop/latency.zip","r") as zipObj:
            fileName=date+"_latency.pickle"
            zipObj.extract(fileName,"C:/Users/CHEN/Desktop/latency/")
        daily_df=pd.read_pickle(file) #load the file
        whole_day=pd.date_range(date+'054500',date+'230000',freq="T")#time of day to be calculated
        #the extra time data is cut at the end 
        output_df=pd.DataFrame({"time":whole_day})

        for start_no in range(len(sensors_short)-1):
            for end_no in range(len(sensors_short)-1,len(sensors_short)): #for each possible combination of sensors in the segment
                st_sensor="01F0"+sensors_short[start_no]+"S"
                name=sensors_short[start_no]+"_"+sensors_short[end_no]
                #create empty columns to be changed 
                output_df[name+"_median_latency_past_"+percentage_list[0]]=""
                output_df[name+"_median_latency_past_"+percentage_list[1]]=""
                output_df[name+"_median_latency_past_"+percentage_list[2]]=""
                output_df[name+"_percent_"+percentage_list[0]]=""
                output_df[name+"_percent_"+percentage_list[1]]=""
                output_df[name+"_percent_"+percentage_list[2]]=""
                
                for t in whole_day:  #calulate past mean speed/latency of each minute
                    for index in range(0,3,1):
                        name2=name+"_median_latency_past_"+percentage_list[index]
                        name3=name+"_percent_"+percentage_list[index]
                        lr_df=pd.read_pickle("C:/Users/CHEN/Desktop/linear_regression_df"+"_"+percentage_list[index]+".pickle") #load a dataframe which contain the slope and intercept of linear regression of each possible segment 
                        #these files will be sent separately (put them in the same folder for the code to work!)
                        #lr_df=pd.read_pickle("C:/Users/homan/Documents/UROP/latency/dataframe/linear_regression_df_with_ti_lr_"+percentage_list[index]+".pickle") (ignore this comment)
                        past=lr_df[name+"_latency"][3]#get the time interval for the segment
                        start_t=t-dt.timedelta(minutes=past)#starting from a past time 
                        end_t=t
                        tot_target=daily_df.loc[(daily_df[st_sensor]>=start_t)&(daily_df[st_sensor]<end_t),name+"_latency"].dropna().copy() #get vehicles between now and now-time interval
                        pass_target=tot_target.loc[tot_target<dt.timedelta(minutes=past)].copy()#get vehicles that pass through the segment within the time interval
                        if len(tot_target)==0:
                            output_df.loc[output_df["time"]==t,[name2]]=np.nan
                            output_df.loc[output_df["time"]==t,[name3]]=np.nan
                            continue
                        else:
                            percent_past=len(pass_target)/len(tot_target)*100 #calculate percentage of vehicles passing within time interval
                            output_df.loc[output_df["time"]==t,[name3]]=percent_past #record the percentage
                        if percent_past>=50: #if more than 50% of vehicle pass, find median in usual way
                            output_df.loc[output_df["time"]==t,[name2]]=np.median(tot_target.dropna())/np.timedelta64(1,'s')
                        elif percent_past>1: #if less than 50% but more than 1% of vehicle pass, find median by slope of linear regression and interval
                            output_df.loc[output_df["time"]==t,[name2]]=percent_past*lr_df[name+"_latency"][0]+lr_df[name+"_latency"][1]
                        else: #if less than 1% of vehicle pass, use "extreme median" value
                            output_df.loc[output_df["time"]==t,[name2]]=lr_df[name+"_latency"][2]
                print(name,w,date,int(time.time()-start_time))

        for start_no_3 in range(len(sensors_short)-1):
            for end_no_3 in range(len(sensors_short)-1,len(sensors_short)): #for each combination
                name_3=sensors_short[start_no_3]+"_"+sensors_short[end_no_3]
                distance=(int(sensors_short[end_no_3])-int(sensors_short[start_no_3]))*100 #find distance between sensors in meter
                for index2 in range(0,3,1):
                    past_name=name_3+"_median_speed_past_"+percentage_list[index2]
                    past_name_2=name_3+"_median_latency_past_"+percentage_list[index2]
                    output_df[past_name]=[(distance/output_df[past_name_2][i])*3.6 for i in output_df.index]  #change latency to speed

        output_df=output_df.loc[15:]
        output_df=output_df.set_index("time")
        print(date,"Done!")
        output_df=output_df[output_df.columns.drop(list(output_df.filter(regex='latency')))]
        output_df.to_pickle("C:/Users/CHEN/Desktop/abc/"+date+"_inputs.pickle")
        #output_df.to_pickle("C:/Users/homan/Documents/UROP/latency/inputs_155-509/"+date+"_inputs_with_ti_lr.pickle")
        os.remove(file)
end_time=time.time()
duration=end_time-start_time #find out runtime 
duration_hour=duration//3600
duration_min=(duration%3600)//60
duration_sec=int((duration%3600)%60)
print(duration_hour,"hour",duration_min,"min",duration_sec,"sec")