import pandas as pd
import numpy as np
import datetime as dt
import os
from zipfile import ZipFile
weekdays=["MON","TUE","WED","THU","FRI"]
all_days=[]
sensors_short=["155","182","248","264","293","339","376","413","467","509"]
interval=5
kkk=pd.DataFrame(data=0,columns=["155","182","248","264","293","339","376","413","467","509"],index=["155","182","248","264","293","339","376","413","467","509"])
for w in range(len(weekdays)):
    partial_days=pd.date_range("20170101","20171231",freq="W-"+weekdays[w])
    partial_days=partial_days.strftime('%Y%m%d')
    all_days.extend(partial_days)
for date in all_days:
    file="C:/Users/CHEN/Desktop/latency/"+date+"_latency.pickle"
    with ZipFile("C:/Users/CHEN/Desktop/latency.zip","r") as zipObj:
        fileName=date+"_latency.pickle"
        zipObj.extract(fileName,"C:/Users/CHEN/Desktop/latency/")
    daily_df=pd.read_pickle(file)
    whole_day=pd.date_range(date+'060000',date+"230000",freq="T")
    for start_no in range(len(sensors_short)-1):
        for end_no in range(start_no+1,len(sensors_short)):
            kkk.loc[kkk.index==sensors_short[start_no], sensors_short[end_no]]+=len(daily_df.loc[daily_df[sensors_short[start_no]+'_'+sensors_short[end_no]+'_latency'].notnull()])
    os.remove(file)
kkk.to_csv("C:/Users/CHEN/Desktop/number_vehicles_complete_journey.csv")
