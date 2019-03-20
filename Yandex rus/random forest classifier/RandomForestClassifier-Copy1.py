#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from  sklearn.metrics import r2_score
from sklearn.metrics import make_scorer


# In[2]:


data = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/random forest classifier/abalone.csv')


# In[3]:


data.head(10)


# In[4]:


data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))


# In[5]:


data.head(10)


# In[6]:


X = data.drop('Rings', axis=1)


# In[7]:


Y = data['Rings']


# In[10]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = list()
for n_estimators in range(1, 51):
    clf = RandomForestRegressor(n_estimators, random_state=1)
    
    score = cross_val_score(clf, X, Y, scoring='r2', cv=k_fold)
    scores.append(score.mean())


# In[33]:


scores = np.array(scores)
scores = scores[scores < 0.52]
print(len(scores)+1)

