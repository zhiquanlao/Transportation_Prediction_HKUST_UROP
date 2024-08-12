import pandas as pd
import numpy as np
import datetime as dt
import time
import os
from zipfile import ZipFile
start_time=time.time()
weekdays=["MON","TUE","WED","THU","FRI"]
all_days=[]
sensors_short=["155","182"]
interval=5
for w in range(len(weekdays)):
    partial_days=pd.date_range("20170101","20171231",freq="W-"+weekdays[w])
    partial_days=partial_days.strftime('%Y%m%d')
    all_days.extend(partial_days)
for date in all_days:
    file="C:/Users/CHEN/Desktop/latency/"+date+"_latency.pickle"
    if os.path.isfile("D:/abc/"+date+"_output_daily_df.pickle"):
        continue
    with ZipFile("C:/Users/CHEN/Desktop/latency.zip","r") as zipObj:
        fileName=date+"_latency.pickle"
        zipObj.extract(fileName,"C:/Users/CHEN/Desktop/latency/")
    daily_df=pd.read_pickle(file)
    whole_day=pd.date_range(date+'060000',date+"230000",freq="T")
    output_daily_df=pd.DataFrame(index=whole_day)
    for start_no in range(len(sensors_short)-1):
        for end_no in range(1,len(sensors_short)):
            if end_no<=start_no:
                continue
            st_sensor="01F0"+sensors_short[start_no]+"S"
            ed_sensor="01F0"+sensors_short[end_no]+"S"
            name=sensors_short[start_no]+"_"+sensors_short[end_no]+"_"
            output_daily_df[name+"median_latency_5"]=""
            #output_daily_df[name+"mean_latency_5"]=""
            daily_df.loc[daily_df[name+"latency"]<=dt.timedelta(0)]=np.nan
            temp_df=daily_df.loc[daily_df[name+"latency"].notnull()]
            temp_df=temp_df.reset_index()
            temp_df=temp_df.drop('index',axis=1)
            dist=(int(name[4:7])-int(name[0:3]))*100
            for ed_time in whole_day:
                st_time=ed_time-dt.timedelta(minutes=interval)
                range_list=temp_df.loc[(temp_df[ed_sensor]>=st_time)&(temp_df[ed_sensor]<=ed_time)]
                
                output_daily_df[name+"median_latency_5"][ed_time]=np.nanmedian(range_list[name+"latency"]/np.timedelta64(1,'s'))
                                #output_daily_df[name+"mean_latency_5"][st_time]=np.mean(range_list[name+"latency"]/np.timedelta64(1,'s'))
            #output_daily_df[name+"mean_speed_5"]=dist/output_daily_df[name+"mean_latency_5"]
            output_daily_df[name+"median_speed_5"]=dist/output_daily_df[name+"median_latency_5"]
            print(date,name)
    output_daily_df.to_pickle("D:/abc/"+date+"_output_daily_df.pickle")
    os.remove(file)
end_time=time.time()
duration=end_time-start_time
hour=duration//3600
minute=(duration%3600)//60
second=int((duration%3600)%60)
print("Run time: ",hour,"hours ",minute,"minutes ",second,"seconds")