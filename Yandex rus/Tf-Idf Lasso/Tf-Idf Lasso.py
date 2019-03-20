#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/Tf-Idf Lasso/salary-train.csv')


# In[2]:


test = pd.read_csv('E:/Prog/ML/Coursera/Yandex rus/Tf-Idf Lasso/salary-test-mini.csv')


# In[3]:


train.head(10)


# In[4]:


train['FullDescription'] = train['FullDescription'].str.lower()


# In[5]:


train.head(10)


# In[6]:


train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVectorizer = TfidfVectorizer(min_df=5)


# In[11]:


X_train_text = TfidfVectorizer.fit_transform(train['FullDescription'])


# In[12]:


train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)


# In[14]:


from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))


# In[16]:


import scipy
X_train = scipy.sparse.hstack([X_train_text, X_train_categ])


# In[17]:


y_train = train['SalaryNormalized']


# In[18]:


from sklearn.linear_model import Ridge
model = Ridge(alpha=1)


# In[19]:


model.fit(X_train, y_train)


# In[22]:


test['FullDescription'] = test['FullDescription'].str.lower()
test['FullDescription'] = test['FullDescription'].str.replace('[^a-zA-Z0-9]', ' ')
X_test_text = TfidfVectorizer.transform(test['FullDescription'])
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = scipy.sparse.hstack([X_test_text, X_test_cat])
y_test = model.predict(X_test)
print(y_test)


# In[ ]:




