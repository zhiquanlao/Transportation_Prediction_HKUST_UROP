import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import mixture

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import datetime
import time
import csv
weekdays=["MON","TUE","WED","THU","FRI"]
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

all_testing_days_lt = [testing_days1, testing_days2, testing_days3, testing_days4, testing_days5, testing_days6, testing_days7, testing_days8, testing_days9, testing_days10, testing_days11, testing_days12, testing_days13, testing_days14, testing_days15, testing_days16, testing_days17, testing_days18, testing_days19, testing_days20]

def sep_weekday(input_df, weekday=[]):
    global df_wd
    df_wd_all = pd.DataFrame()
    
    for day in weekday:
        df_wd = input_df.loc[input_df.index.weekday == day]
        df_wd_all = pd.concat([df_wd_all, df_wd])
    return df_wd_all

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

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
df = pd.read_pickle('C:/Users/CHEN/Desktop/UROP_data/df_new_ver5_20190401.pickle')    # import the dataframe
start_time=time.time()
# Each row represents the time. Columns are the input features and the target. 

month = 12
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

# Select 2017 data
df = df.loc[df.index.year == 2017]
df = df.loc[df.index.month <= month]


start="339"
end="509"
name='median_latency'
target_columns = [name]
#name=starting point only,actually is a segment from it to next detector
#end point of past latency=509
#acc=accumulation, en=entry rate, ex=exit rate, in=in flux, out=out flux, p1=past 1 min

#############################################################################################
input_df=pd.read_pickle('C:/Users/CHEN/Desktop/UROP_data/all_in_out/all_in_out.pickle')
pldf=pd.read_pickle('C:/Users/CHEN/Desktop/UROP_data/pl_inputs.pickle')
#output_df=pd.read_pickle('C:/Users/CHEN/Desktop/UROP_data/all_out_s.pickle')
output_df=pd.DataFrame()
output_df_date=pd.DataFrame()
for day in range(5):
    all_days=pd.date_range("20170101","20171231",freq="W-"+weekdays[day])
    all_days=all_days.strftime('%Y%m%d')
    
    for date in all_days:
        path="C:/Users/CHEN/Desktop/UROP_data/Output_daily_zip/"+date+"_output_daily_df.pickle"
        daily_df=pd.read_pickle(path)
        daily_df=daily_df[daily_df.index.hour!=23]
        daily_df=daily_df.reset_index()
        output_df_date=pd.concat([output_df_date,daily_df['index']],ignore_index=True)
        daily_df=daily_df['339_509_median_latency_5']
        output_df=pd.concat([output_df,daily_df],ignore_index=True)
    #output_df=output_df.rename(columns={0:"339_509_median_latency_5"})
    output_df=output_df[~output_df.index.duplicated()]
    #output_df_date=output_df_date.rename(columns={0:"date_time"})
    output_df_date=output_df_date[~ output_df_date.index.duplicated()]
    output_df["date_time"]=output_df_date[0]
    output_df=output_df.set_index("date_time")
df['median_latency']=output_df[0]
#df['median_latency']=output_df[start+'_'+end+'_median_latency_5']
#df['mean_latency']=output_df[start+'_'+end+'_mean_latency_5']
df[start+"_"+end+"median_latency_regression"]=pldf[start+"_"+end+"_p_l_median"]
sensors=['339','376','413','467','509']
start_no=sensors.index(start)
end_no=sensors.index(end)
input_columns_new=['339_out_p1','339_out_p2','339_out_p3','413_out_p1','413_out_p2','413_out_p3',"acc_tot","413_acc",start+"_"+end+"median_latency_regression"]
for list_no in range(start_no+1,end_no+1):
    for past in range (5,16,5):
        if list_no-start_no>2 and past==5:
            continue
        else:
            name_1=start+"_"+sensors[list_no]+'_median_speed_past_'+str(past)
            df[name_1]=input_df[name_1]
            input_columns_new.append(name_1)
if end_no-start_no>1:
    for past in range (5,16,5):
        name_2=sensors[start_no+1]+"_"+end+'_median_speed_past_'+str(past)
        input_columns_new.append(name_2)
        df[name_2]=input_df[name_2]

#############################################################################################
MSE_test_list=[]
MAPE_test_list=[]
col_list=[]
for col in df.columns:
    col_list.append(col)
MSE_raw=[2613,3413,13026,23910,5367]
MAPE_raw=[2.435,2.737,3.334,4.066,3.411]
"""
GMM_features_list=[['339_in','376_out','467_out','339_out','413_ex','467_ex','467_in','339_in_p1','467_ex','467_out_p1','413_ex_p1','339_in_p2','467_in_p1','413_in','467_ex_p1','413_in','467_ex_p1','339_out_p1','413_out','413_out_p1','339_out_p2','376_out_p1','413_en'],
                   ['339_in','339_en','339_ex','376_out','339_in','339_out','467_in','376_in','467_out','413_ex','413_en','413_out','376_out_p1','376_in_p1','413_acc','413_in','339_en_p1','339_ex_p1','339_acc_p1','339_out_p1'],
                   ['339_509median_latency_regression','467_ex_p4','339_ex_p4','339_in_p5','467_en_p4','467_ex_p3','339_in_p4','past_20','en_tot','467_ex_p5','467_out_p4','467_acc_p4','acc_tot','467_en_p3','5ma_latency','467_ex_p2','467_en_p5','past_30','ex_tot','339_out_p4','467_in_p4'],
                   ['467_out_p5','413_out_p5','ex_tot','467_ex_p5','413_in_p5','467_in_p5','413_ex_p5','376_ex_p5','467_en_p5','376_out_p5','376_en_p5','413_out_p4','413_acc_p5','413_in_p4','467_ex_p4','376_acc_p5','413_en_p5','339_in_p5','467_en_p4'],
                   ['467_out','467_out_p2','467_in','376_out_p3','467_in_p2','413_ex_p2','376_out_p2','339_509median_latency_regression','413_in_p3','376_in_p3','413_en_p2','376_out','413_out_p2','413_out_p3','376_ex_p3','376_in_p2','467_out_p1','339_out_p3','376_out_p4','467_ex_p2']
                   ]

GFL=[['467_in_p1','339_in_p1','467_ex','376_out','413_in','467_ex'],
                   ['413_acc','339_en_p1','339_in','376_out_p1','376_in','376_out'],
                   ['467_in_p4','en_tot','467_en_p4','5ma_latency','acc_tot','ex_tot'],
                   ['413_out_p5','467_in_p5','413_ex_p5','413_in_p5','376_out_p5','467_out_p5'],
                   ['376_in_p2','467_in_p2','467_out_p2','376_ex_p3','339_out_p3','413_ex_p2']
    ]

GNL=[[3.75,5.75,5.75,5.45,4.85,5.45],
                     [4.15,4.05,4.15,4.15,4.75,4.75],
                     [3.2,5.55,6.3,4.65,6.25,4.65],
                     [5.3,5,5.55,4.9,5.25,5.15],
                     [5.65,5.6,5.65,6,6.05,6]
    ]

GMM_features_list=[[['376_out'],['467_ex'],['467_ex','376_out'],['339_in_p1','467_ex','376_out'],['413_in'],['467_ex','376_out','413_in'],['339_in_p1','467_ex','376_out','413_in'],['339_in_p1','376_out','413_in','467_ex'],['339_in_p1','467_ex'],['339_in_p1','376_out','413_in'],['339_in_p1','467_ex','376_out','413_in'],['376_out','413_in','467_ex'],['467_in_p1','467_ex','376_out','413_in'],['376_out','413_in']],
                   [['339_in'],['339_en_p1','376_out_p1'],['376_out_p1','376_in'],['376_in','376_out'],['376_in'],['339_en_p1','376_out'],['339_in','376_out_p1','376_in','376_out'],['413_acc','339_en_p1','339_in'],['376_out_p1'],['376_out'],['413_acc','339_in','376_out'],['339_en_p1','376_in'],['413_acc','339_en_p1','376_in'],['339_in','376_out_p1','376_in'],['413_acc','339_en_p1'],['339_en_p1'],['376_out_p1','376_out'],['413_acc','339_in'],['339_en_p1','339_in']],
                   [['5ma_latency'],['5ma_latency','acc_tot'],['467_in_p4','en_tot','467_en_p4','ex_tot'],['en_tot','467_en_p4','ex_tot'],['en_tot','467_en_p4','ex_tot'],['467_in_p4','467_en_p4'],['467_in_p4','ex_tot'],['467_en_p4','5ma_latency'],['467_in_p4','467_en_p4','ex_tot'],['en_tot','467_en_p4'],['en_tot','467_en_p4','5ma_latency','ex_tot'],['467_en_p4','acc_tot','ex_tot'],['467_en_p4','ex_tot'],['467_en_p4'],['467_in_p4','5ma_latency','acc_tot'],['467_in_p4','467_en_p4','acc_tot','ex_tot'],['467_in_p4','acc_tot'],['en_tot','5ma_latency','acc_tot'],['en_tot','5ma_latency','ex_tot']],
                   [['413_in_p5'],['376_out_p5'],['413_ex_p5','413_in_p5','467_out_p5'],['413_out_p5','413_ex_p5','467_out_p5'],['413_out_p5','467_in_p5','376_out_p5','467_out_p5'],['467_in_p5','413_ex_p5','413_in_p5','467_out_p5'],['413_ex_p5'],['413_out_p5','467_in_p5','413_ex_p5','413_in_p5'],['413_out_p5','376_out_p5','467_out_p5'],['413_out_p5','467_in_p5','413_ex_p5','376_out_p5'],['413_out_p5','413_ex_p5','413_in_p5','467_out_p5'],['413_ex_p5','413_in_p5','376_out_p5'],['413_out_p5','467_in_p5','413_in_p5'],['413_out_p5','467_in_p5','413_ex_p5','413_in_p5','467_out_p5'],['413_out_p5','467_in_p5','413_ex_p5'],['413_out_p5','413_in_p5','376_out_p5'],['413_out_p5','413_ex_p5'],['413_out_p5','467_in_p5','376_out_p5'],['413_out_p5'],['413_out_p5','467_in_p5','467_out_p5']],
                   [['376_in_p2'],['413_ex_p2'],['376_ex_p3'],['376_in_p2','467_out_p2','376_ex_p3','339_out_p3'],['467_in_p2','467_out_p2','376_ex_p3','339_out_p3','413_ex_p2'],['376_in_p2','376_ex_p3','339_out_p3','413_ex_p2'],['376_in_p2','467_in_p2','467_out_p2','376_ex_p3','339_out_p3','413_ex_p2'],['467_in_p2','339_out_p3','413_ex_p2'],['467_in_p2','339_out_p3'],['339_out_p3','413_ex_p2'],['376_in_p2','467_in_p2','376_ex_p3','339_out_p3'],['376_in_p2','467_in_p2','467_out_p2','376_ex_p3','413_ex_p2'],['376_in_p2','413_ex_p2'],['376_in_p2','467_out_p2','376_ex_p3','339_out_p3','413_ex_p2'],['376_in_p2','467_in_p2','467_out_p2','376_ex_p3','339_out_p3'],['376_in_p2','376_ex_p3'],['467_in_p2','467_out_p2','376_ex_p3','413_ex_p2'],['376_in_p2','376_ex_p3','339_out_p3']]
    ]
GMM_no_cluster_list=[[5.45,5.75,5.6,5.649999,4.85,5.349999,5.449999,5.449999,5.75,5.349999,5.449999,5.35,4.949999,5.15],
                     [4.15,4.1,4.45,4.75,4.75,4.75,4.4,4.45,4.11666666,4.15,4.75,4.35,4.4,4.316666,4.35,4.1,4.05,4.45,4.15,4.1],
                     [4.65,5.45,4.92,5.5,5.5,4.75,3.92,5.475,4.716666666,5.925,5.2875,5.7333333,5.475,6.3,4.7,5.1,4.725,5.483333333,4.95],
                     [4.9,5.25,5.2,5.33333,5.175,5.15,5.55,5.1875,5.233333,5.275,5.225,5.233333,5.0666666,5.18,5.2833333333,5.18,5.2833333333,5.1499999999,5.425,5.1833333333,5.3,5.15],
                     [5.65,6,6,5.8375,5.86,5.925,5.825,5.883333333,5.824999999,6.025,5.825,5.779999999,5.825,5.87,5.79,5.825,5.8125,5.899999999]
    ]

for i in range(len(GMM_features_list)):
    for lol in GMM_features_list[i]:
        avg=0
        for e in lol:
            for j in range(len(GFL[i])):
                if GFL[i][j]==e:
                    avg+=GNL[i][j]
                    break
        GMM_no_cluster_list[i].append(avg/len(lol))
        """
GMM_features_list=[[['339_467_median_speed_past_15', '376_509_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_15'],
['339_467_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['339_467_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['376_509_median_speed_past_15', '339_467_median_speed_past_10', '376_509_median_speed_past_10'],
['376_509_median_speed_past_15'],
['339_509_median_speed_past_15'],
['376_509_median_speed_past_15', '339_467_median_speed_past_10'],
['376_509_median_speed_past_15', '339_413_median_speed_past_10'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_467_median_speed_past_15', '376_509_median_speed_past_15', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_10'],
['339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_413_median_speed_past_15'],
['339_467_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
],
                   [['376_509_median_speed_past_15', '339_467_median_speed_past_15'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['376_509_median_speed_past_15'],
['376_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['376_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15'],
['376_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15']
],
                   [['376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['339_467_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_10', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_10'],
['376_509_median_speed_past_15', '339_467_median_speed_past_10', '376_509_median_speed_past_10'],
['376_509_median_speed_past_15', '339_467_median_speed_past_10', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10', '376_509_median_speed_past_10'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15'],
['376_509_median_speed_past_10'],
['376_509_median_speed_past_15'],
['376_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10']
],
                   [['376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_15', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_10'],
['339_413_median_speed_past_15', '339_413_median_speed_past_10']
],
                   [['339_509_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15', '339_467_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_15'],
['376_509_median_speed_past_10', '339_467_median_speed_past_10', '339_413_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '339_467_median_speed_past_10'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '376_509_median_speed_past_15', '339_467_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_10', '339_413_median_speed_past_15'],
['339_509_median_speed_past_15', '339_413_median_speed_past_10', '339_413_median_speed_past_15'],
['376_509_median_speed_past_15', '339_467_median_speed_past_15', '339_467_median_speed_past_10', '339_413_median_speed_past_10'],
['339_509_median_speed_past_15', '339_467_median_speed_past_15', '339_413_median_speed_past_15'],
]
    ]
def avg(li):
    return sum(li)/len(li)
MSE_of_features=[]
MSE_of_features.append([])
MSE_of_features.append([])
MSE_of_features.append([])
MSE_of_features.append([])
for w in range(5):
        MSE_test_list.append([])
        MAPE_test_list.append([])
input_feature=[]
n_clusters=6
"""
def traverse(number,w):
    k=f'{number:06b}'
    a=[]
    avg=0
    for i in range(len(k)):
        if k[i]=='1':
            a.append(GMM_features_list[w][i])
            avg+=GMM_no_cluster_list[w][i]
    return a,round(avg/len(a))
"""
input_feature=['acc_tot','467_out']

BIC=[[[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]]
     ]
AIC=[[[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]],
     [[],[],[],[],[],[],[],[],[],[]]
     ]
for w in range(5):
    #for j in range(len(GMM_features_list[w])):
        n_best=1
        MSE_best=9999999
        for n_clusters in range(1,11):
            MSE_avg=0
            for td in range(20):
                df_wd = sep_weekday(df, [w])
                df_wd=df_wd[list(set(input_columns_new+['median_latency']+input_feature))]
                df_train_or, df_test_or = sep_train_test(df_wd, weekday=[w], testing_days=all_testing_days_lt[td])
                df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
                df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
                
                df_train = df_train_or.dropna()
                df_test = df_test_or.dropna()
                
                #n_clusters = 6 # Separating to how many clusters
                # Various Setting can be tested 
                
                gmm_n = mixture.GaussianMixture(n_components=n_clusters, max_iter=500)
                df_train_gmm_n = df_train[input_feature]    # these array is the input for the GMM to do clustering
                scaler_gmm_n = StandardScaler()
                scaler_gmm_n.fit(df_train_gmm_n)
                
                df_train_gmm_n = scaler_gmm_n.transform(df_train_gmm_n)
                gmm_n.fit(df_train_gmm_n)    # train the GMM by the training set
                """
                #######calculating BIC#############
                n_par=n_clusters-1+n_clusters*len(input_feature)+n_clusters*len(input_feature)*(len(input_feature)+1)//2
                max_log=gmm_n.score(df_train_gmm_n)*len(df_train_gmm_n)
                BIC[w][n_clusters-1].append(np.log(len(df_train_gmm_n))*n_par-2*max_log)
                AIC[w][n_clusters-1].append(2*n_par-2*max_log)
for w in range(5):
    for j in range(len(BIC[w])):
        BIC[w][j]=avg(BIC[w][j])
        AIC[w][j]=avg(AIC[w][j])
for w in range(5):
    plt.plot([1,2,3,4,5,6,7,8,9,10],BIC[w])
    plt.savefig("C:/Users/CHEN/Desktop/"+weekdays[w]+str(input_feature)+"_BIC_clusters.png")
    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],AIC[w])
    plt.savefig("C:/Users/CHEN/Desktop/"+weekdays[w]+str(input_feature)+"_AIC_clusters.png")
    plt.clf()
    
"""
                # Adding a new column for the predicted group
                df_train['GMM'] = gmm_n.predict(df_train_gmm_n)
                
                ######## Testing
                #gmm_n = mixture.GaussianMixture(n_components=n_clusters, max_iter=500)
                df_test_gmm_n = df_test[input_feature]    # these array is the input for the GMM to do clustering
                #scaler_gmm_n = StandardScaler()
                #scaler_gmm_n.fit(df_test_gmm_n)
                df_test_gmm_n = scaler_gmm_n.transform(df_test_gmm_n)
                #gmm_n.fit(df_test_gmm_n)   
                df_test['GMM'] = gmm_n.predict(df_test_gmm_n)
                
                df_train_pred  = pd.DataFrame()
                df_test_pred   = pd.DataFrame()
                GMM(n_clusters)
                
                df_train_pred = df_train_pred.sort_index()
                df_test_pred = df_test_pred.sort_index()
                
                df_train['pred_new'] = df_train_pred['pred_new']
                df_test['pred_new']  = df_test_pred['pred_new']
                
                #MSE_train = mean_squared_error(df_train['median_latency'], df_train['pred_new'])
                #MAPE_train = MAPE(df_train['median_latency'], df_train['pred_new'])
                MSE_test = mean_squared_error(df_test['median_latency'], df_test['pred_new'])
                MSE_avg+=MSE_test
            MSE_avg=MSE_avg/20.0
            if MSE_avg<MSE_best:
                MSE_best=MSE_avg
                n_best=n_clusters
        MSE_of_features[0].append(w)
        MSE_of_features[1].append(input_feature)
        MSE_of_features[2].append(MSE_best)
        MSE_of_features[3].append(n_best)
        #MAPE_test = MAPE(df_test['median_latency'], df_test['pred_new'])
        #MSE_test_list[w].append(MSE_test)
        """    
    #for w in range(5):    
        #print("weekday: {}, MSE: {}, MAPE: {}".format(w,avg(MSE_test_list[w]),avg(MAPE_test_list[w])))
        #MSE_of_features[0].append(w)
        #MSE_of_features[1].append(col)
        #MSE_of_features[2].append(avg(MSE_test_list[w]))
#MSE_df=pd.DataFrame(MSE_of_features)
#MSE_df=MSE_df.T
#MSE_df.to_pickle("C:/Users/CHEN/Desktop/MSE_of_GMM_multi_features_mutual_info.pickle")
#MSE_df.to_csv("C:/Users/CHEN/Desktop/MSE_of_GMM_multi_features_mutual_info.csv")



    #print("weekday: {}, MSE: {}, MAPE: {}".format(w,avg(MSE_test_list[w]),avg(MAPE_test_list[w])))
mutual_info_features=[['339_509_median_speed_past_15','339_467_median_speed_past_15','376_509_median_speed_past_15','339_467_median_speed_past_10','376_509_median_speed_past_10','339_413_median_speed_past_15','339_413_median_speed_past_10'],
                      ['339_509_median_speed_past_15','376_509_median_speed_past_15','339_467_median_speed_past_15','339_467_median_speed_past_10','376_509_median_speed_past_10','339_413_median_speed_past_15','339_413_median_speed_past_10'],
                      ['339_509_median_speed_past_15','339_467_median_speed_past_15','376_509_median_speed_past_15','339_467_median_speed_past_10','376_509_median_speed_past_10','339_413_median_speed_past_15','339_413_median_speed_past_10'],
                      ['339_509_median_speed_past_15','376_509_median_speed_past_15','339_467_median_speed_past_15','376_509_median_speed_past_10','339_467_median_speed_past_10','339_413_median_speed_past_15','339_413_median_speed_past_10'],
                      ['339_509_median_speed_past_15','376_509_median_speed_past_15','339_467_median_speed_past_15','376_509_median_speed_past_10','339_467_median_speed_past_10','339_413_median_speed_past_10','339_413_median_speed_past_15']
    ]
for i in range(len(mutual_info_features)):
    for j in range(len(mutual_info_features[i])):
        if mutual_info_features[i][j] not in col_list:
            print("i:{}, j: {}, col: {}".format(i,j,mutual_info_features[i][j]))
def traverse(number,w):
    k=f'{number:07b}'
    a=[]
    for i in range(len(k)):
        if k[i]=='1':
            a.append(mutual_info_features[w][i])
    return a
mutual_info_MSE=[[],
                 [],
                 []
    ]
for w in range(1,5):
    for no in range(1,2**len(mutual_info_features[w])):
        input_feature=traverse(no,w)
        MSE_avg=0
        for td in range(20):
            df_wd = sep_weekday(df, [w])
            df_wd=df_wd[list(set(input_columns_new+['median_latency']+input_feature))]
            df_train_or, df_test_or = sep_train_test(df_wd, weekday=[w], testing_days=all_testing_days_lt[td])
            df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
            df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
            
            df_train = df_train_or.dropna()
            df_test = df_test_or.dropna()
            
            n_clusters = 6 # Separating to how many clusters
            # Various Setting can be tested 
            
            gmm_n = mixture.GaussianMixture(n_components=n_clusters, max_iter=500)
            df_train_gmm_n = df_train[input_feature]    # these array is the input for the GMM to do clustering
            scaler_gmm_n = StandardScaler()
            scaler_gmm_n.fit(df_train_gmm_n)
            
            df_train_gmm_n = scaler_gmm_n.transform(df_train_gmm_n)
            gmm_n.fit(df_train_gmm_n)    # train the GMM by the training set
            
            # Adding a new column for the predicted group
            df_train['GMM'] = gmm_n.predict(df_train_gmm_n)
            
            ######## Testing
            #gmm_n = mixture.GaussianMixture(n_components=n_clusters, max_iter=500)
            df_test_gmm_n = df_test[input_feature]    # these array is the input for the GMM to do clustering
            #scaler_gmm_n = StandardScaler()
            #scaler_gmm_n.fit(df_test_gmm_n)
            df_test_gmm_n = scaler_gmm_n.transform(df_test_gmm_n)
            #gmm_n.fit(df_test_gmm_n)   
            df_test['GMM'] = gmm_n.predict(df_test_gmm_n)
            
            df_train_pred  = pd.DataFrame()
            df_test_pred   = pd.DataFrame()
            GMM(n_clusters)
            
            df_train_pred = df_train_pred.sort_index()
            df_test_pred = df_test_pred.sort_index()
            
            df_train['pred_new'] = df_train_pred['pred_new']
            df_test['pred_new']  = df_test_pred['pred_new']
            
            #MSE_train = mean_squared_error(df_train['median_latency'], df_train['pred_new'])
            #MAPE_train = MAPE(df_train['median_latency'], df_train['pred_new'])
            MSE_test = mean_squared_error(df_test['median_latency'], df_test['pred_new'])
            MSE_avg+=MSE_test
        MSE_avg=MSE_avg/20.0
        mutual_info_MSE[0].append(w)
        mutual_info_MSE[1].append(input_feature)
        mutual_info_MSE[2].append(MSE_avg)
mutual_MSE=pd.DataFrame(mutual_info_MSE)
mutual_MSE=mutual_MSE.T
mutual_MSE.to_pickle("C:/Users/CHEN/Desktop/mutual_MSE_of_GMM_six_clusters.pickle")
mutual_MSE.to_csv("C:/Users/CHEN/Desktop/mutual_MSE_of_GMM_six_clusters.csv")
"""


