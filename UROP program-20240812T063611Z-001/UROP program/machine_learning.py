import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import mixture

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import datetime
import time
import csv
import math
#####linear regression#####
"""
df=pd.read_csv("C:/Users/CHEN/Desktop/data3.0.csv")
X=df.iloc[:,7:9]
y=df.iloc[:,9]
y[y=='是']=1
y[y=='否']=0
w1,w2,beta=random.random(),random.random(),random.random()
def cost_function():
    ans=0
    for i in range(len(y)):
        if y.iloc[i]==1:
            ans=ans+np.log(math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta)/(1+math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta)))
        else:
            ans=ans+np.log(1/(1+math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta)))
    return ans
def gradient():
    gra=[]
    ans=0
    for i in range(len(y)):
        ans=ans+X.iloc[i,0]/(1+math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta))-X.iloc[i,0]*y.iloc[i]
    gra.append(ans)
    ans=0
    for i in range(len(y)):
        ans=ans+X.iloc[i,1]/(1+math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta))-X.iloc[i,1]*y.iloc[i]
    gra.append(ans)
    ans=0
    for i in range(len(y)):
        ans=ans+1/(1+math.exp(w1*X.iloc[i,0]+w2*X.iloc[i,1]+beta))-y.iloc[i]
    gra.append(ans)
    return gra
learn_rate=0.3
for i in range(3000):
    gra=gradient()
    w1=w1-gra[0]
    w2=w2-gra[1]
    beta=beta-gra[2]
plt.scatter(X.iloc[:,0],X.iloc[:,1])
##################
"""

######decision tree###############
"""
class node:
    def __init__(self):
        self.name=None
        self.subtree={}
        self.isLeaf=None
        self.category=None
class DecisionTree:
    def __init__(self,criterion,pruning=None):
        self.criterion=criterion
        self.pruning=pruning
    def entropy(self,y):
        p=pd.value_counts(y)
        return np.sum(-p*np.log2(p))
    def infoGain(self,X, y,a):
        #X.iloc[:,a]可以获得整个dataframe
        ent=self.entropy(y)
        lol=X.iloc[:,a]
        uni=pd.unqique(lol)
        s=0
        for val in uni:
            s+=len(lol[lol==val])/len(lol)*self.entropy(lol)
        return s
    def feature_infoGain(self,X,y):
        feature=0
        large=-10000
        for a in range(len(X.column)):
            info_gain=self.infoGain(X, y, a)
            if info_gain>large:
                large=info_gain
                feature=a
        return feature

    def generate_tree(self,X,y):
        mytree=node()
        if y.nunique()==1:
            mytree.isLeaf=True
            mytree.category=y.iloc[0]
            return mytree
        if X.empty:
            mytree.isLeaf=True
            mytree.category=pd.value_counts(y).index[0]
            return mytree
        feature=self.feature_infoGain(X, y)
        mytree.name=X.columns[feature]
        uni=pd.unique(X.iloc[:,feature])
        sub_X=X.drop(X.columns[feature],axis=1)
        for val in uni:
            mytree[val]=self.generate_tree(sub_X[X.iloc[:,feature]==val],y[X.iloc[:,feature]==val])
        return mytree
    def fit(self,X_train,y_train):
        self.tree=self.generate_tree(X_train,y_train)
        return self
    def predict_single(self,X):
        nod=self.tree
        while nod.isLeaf==False:
            nod=nod.subtree[X[nod.name]]
        return nod.category
    def predict(self,X):
        ans=[]
        for i in range(len(X)):
            ans.append(self.predict_single(X.iloc[[i]]))
        return pd.Series(ans)
    """
###########perceptron######################
class perceptron:
    def __init__(self,eta=0.1):
        self.eta=eta
    def f(x):
        if x>=0:
            return 1
        else:
            return 0
    def update(self,x,y):
        no_change=0
        for i in range(len(x)):
            if y-self.f(np.dot(self.w,x))!=0:
                no_change+=1
                self.w[i]=self.w[i]+self.eta*(y-self.f(np.dot(self.w,x)))*x[i]
        if no_change==0:
            return False
        else:
            return True
    def fit(self,x,y):
        self.w=np.zeros((1,len(x)))
        self.b=random.random()
        for i in range(len(self.w)):
            self.w[i]=random.random()
        while self.update(x,y):
            x=x
        return self
    def predict(self,x):
        return self.f(np.dot(self.w,x))
################backpropagation###############################
class BpNN(object):
    def __init__(self,no_layer,alpha=0.01):
        self.alpha=alpha
        self.no_layer=no_layer
    def sigmoid(z):
        return 1/(1+math.exp((-1)*z))

    def fit(self,X,y):
        self.w=np.zeros((len(y),self.no_layer))
        