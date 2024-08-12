# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:16:49 2022

@author: CHEN
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

import datetime
import time
import csv
import matplotlib.pyplot as plt
df=pd.read_pickle("C:/Users/CHEN/Desktop/155_182_current_all.pickle")
df=df[['155_182_acc','155_182_median_latency_5','155_182_p_l_median']]
mon=df[df.index.weekday==0]
tue=df[df.index.weekday==1]
wed=df[df.index.weekday==2]
thu=df[df.index.weekday==3]
fri=df[df.index.weekday==4]

mon=mon.groupby([mon.index.hour,mon.index.minute]).agg('mean')
tue=tue.groupby([tue.index.hour,tue.index.minute]).agg('mean')
wed=wed.groupby([wed.index.hour,wed.index.minute]).agg('mean')
thu=thu.groupby([thu.index.hour,thu.index.minute]).agg('mean')
fri=fri.groupby([fri.index.hour,fri.index.minute]).agg('mean')
"""
tim=[]
for i in range(1,1022):
    tim.append(i)
tim=np.array(tim)
model1=GPR()
model2=GPR()
model3=GPR()
model4=GPR()
model5=GPR()
model1.fit(tim,mon['155_182_median_latency_5'])
model2.fit(tim,tue['155_182_median_latency_5'])
model3.fit(tim,wed['155_182_median_latency_5'])
model4.fit(tim,thu['155_182_median_latency_5'])
model5.fit(tim,fri['155_182_median_latency_5'])
result=pd.DataFrame()
result['0']=model1.predict(tim)
result['1']=model2.predict(tim)
result['2']=model3.predict(tim)
result['3']=model4.predict(tim)
result['4']=model5.predict(tim)
"""