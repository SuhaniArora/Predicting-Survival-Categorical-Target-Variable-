#%%

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
import numpy as np
from pandas import get_dummies
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
import time
import sys
import os


#%%

DeprecationWarning('ignore')
warnings.filterwarnings('ignore')

#%%

df=pd.read_csv("titanic.csv")

#%%

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=112)
del df

#%%

train.isnull().sum()

#%%

def drop(train):
    train.drop(['Cabin'],axis=1,inplace=True)
    train.drop(['PassengerId'],axis=1,inplace=True)
    train.drop(['Ticket'],axis=1,inplace=True)
    train.drop(['Name'],axis=1,inplace=True)
    train.Age.fillna(29,inplace=True)
    train.Embarked.fillna('S',inplace=True)
    train=pd.get_dummies(train,drop_first=True)
    return train


#%%

train=drop(train)

#%%

train.head()

#%%

def x_and_y(df):
    x=df.drop(['Survived'],axis=1)
    y=df['Survived']
    return x,y
x_train,y_train=x_and_y(train)

#%%

log_model=LogisticRegression()


#%%

log_model.fit(x_train,y_train)


#%%

prediction=log_model.predict(x_train)


#%%

score=accuracy_score(y_train,prediction)
score

#%%

test=drop(test)
x_test,y_test=x_and_y(test)

#%%

prediction1=log_model.predict(x_test)

#%%

score1=accuracy_score(y_test,prediction1)
score1

#%%

dec=DecisionTreeClassifier(criterion = 'entropy', max_depth = 10 ,min_samples_leaf = 5)

#%%

dec.fit(x_train,y_train)

#%%

train_prediction = dec.predict(x_train)

#%%

score1 = accuracy_score(y_train,train_prediction)
score1

#%%

test_prediction = dec.predict(x_test)

#%%

score1 = accuracy_score(y_test,test_prediction)
score1

#%%

rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(x_train, y_train)
train_prediction = rfc.predict(x_train)
score1 = accuracy_score(y_train,train_prediction)
score1

#%%

test_prediction = rfc.predict(x_test)
score1 = accuracy_score(y_test,test_prediction)

#%%
score1