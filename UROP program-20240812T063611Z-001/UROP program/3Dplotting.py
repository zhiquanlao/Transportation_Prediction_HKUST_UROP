
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
speed=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_speed.pickle")
df=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/new_output_df_155-509.pickle")
past_speed=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_182_speed_congest.pickle")
out_acc=pd.read_pickle("C:/Users/CHEN/Desktop/urop data/155_264_out_acc.pickle")
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(out_acc['155_in'],out_acc['155_182_acc'],df['155_182_median_latency_5'],alpha=0.01)
plt.show()
