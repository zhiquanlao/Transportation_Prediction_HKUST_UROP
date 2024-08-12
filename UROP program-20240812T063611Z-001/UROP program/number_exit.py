# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:36:03 2022

@author: CHEN
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from zipfile import ZipFile
weekdays=["MON","TUE","WED","THU","FRI"]
all_days=[]
sensors_short=["155","182","248","264","293","339","376","413","467","509"]
interval=5
kkk=pd.DataFrame(data=0,columns=["182_ex","248_ex","264_ex","293_ex","339_ex","376_ex","413_ex","467_ex"],index=["lol"])
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
    for ed_sensors in range(1,len(sensors_short)-1):
        pre='01F0'+sensors_short[ed_sensors-1]+'S'
        now='01F0'+sensors_short[ed_sensors]+'S'
        post='01F0'+sensors_short[ed_sensors+1]+'S'

        kkk.loc[kkk.index=='lol',sensors_short[ed_sensors]+'_ex']
    os.remove(file)