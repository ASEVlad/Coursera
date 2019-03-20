#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[35]:


df1 = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/metrics for classification/scores.csv')


# In[36]:


df1.head(10)


# In[ ]:





# In[ ]:





# In[2]:


df = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/metrics for classification/classification.csv')


# In[3]:


df.info()


# In[19]:


df.head(10)


# In[29]:


TP = len(df[(df['true'] == 1) & (df['pred'] == 1)])
FP = len(df[(df['true'] == 1) & (df['pred'] == 0)])
TN = len(df[(df['true'] == 0) & (df['pred'] == 0)])
FN = len(df[(df['true'] == 0) & (df['pred'] == 1)])
print(len(df))
print(TP, FP, TN, FN)


# In[58]:


from sklearn import metrics


# In[33]:


score = list()
score.append(metrics.accuracy_score(df['true'], df['pred']))
score.append(metrics.precision_score(df['true'], df['pred']))
score.append(metrics.recall_score(df['true'], df['pred']))
score.append(metrics.f1_score(df['true'], df['pred']))
score


# In[40]:


score1 = list()
score1.append(metrics.roc_auc_score(df1['true'], df1['score_logreg']))
score1.append(metrics.roc_auc_score(df1['true'], df1['score_svm']))
score1.append(metrics.roc_auc_score(df1['true'], df1['score_knn']))
score1.append(metrics.roc_auc_score(df1['true'], df1['score_tree']))
score1


# In[57]:


res = list()
for col in df1.columns:
    mas = metrics.precision_recall_curve(df1['true'], df1[col])
    data = pd.DataFrame({'precision' : mas[0], 'recall' : mas[1]})
    recall = data[data['recall'] > 0.7]['precision'].max()
    res.append(recall)
print(res)


# In[46]:


recall


# In[ ]:




