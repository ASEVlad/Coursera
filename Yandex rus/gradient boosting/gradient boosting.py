#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from math import exp
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# In[3]:


data = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/gradient boosting/gbm-data.csv')


# In[16]:


y = data.iloc[:, 0].ravel()
X = data.iloc[:, 1:]


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# In[18]:


vfunc = np.vectorize(lambda y: 1/(1+exp(-y)))
def calc(X,y_true,n_est):
    score=np.empty(n_est)
    df=clf.staged_decision_function(X)
    for i, y_pred in enumerate(df):
        score[i] = log_loss(y_true, vfunc(y_pred))
    return score


# In[22]:


learning_rate=[1, 0.5, 0.3, 0.2, 0.1]
for lr in learning_rate:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = lr)
    clf.fit(X_train, y_train)
    
    train_score = calc(X_train, y_train, 250)
    test_score = calc(X_test, y_test, 250)
    
    if lr==0.2: test_score02=test_score
    
    plt.figure()
    plt.plot(test_score, 'r', linewidth=3)
    plt.plot(train_score, 'g', linewidth=2)
    plt.legend(['test', 'train'])


# In[21]:


#underfitting


# In[23]:


test_scorePD=pd.DataFrame(data=test_score02)
test_scorePD.sort_values([0],inplace=True, ascending=True)
test_scorePD[0]=test_scorePD[0].map(lambda x: round(x,2))
print('Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается, при learning_rate = 0.2')
val=test_scorePD.head(1)
print(val)


# In[24]:


print(test_score02)


# In[25]:


clf = RandomForestClassifier(n_estimators=val.index.values[0],random_state=241)#Конструктор случайного леса
clf.fit(X_train,y_train)

pred = clf.predict_proba(X_test)
score=log_loss(y_test, pred)

print('Какое значение log-loss на тесте получается у этого случайного леса?')
print(round(score,2))


# In[ ]:




