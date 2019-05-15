#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[11]:


import seaborn as sns


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df=pd.read_csv('kyphosis.csv')


# In[8]:


df.head()


# In[9]:


df.info


# In[12]:


sns.pairplot(df, hue='Kyphosis')


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X=df.drop('Kyphosis', axis=1)


# In[16]:


y=df['Kyphosis']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


dtree=DecisionTreeClassifier()


# In[27]:


dtree.fit(X_train,y_train)


# In[28]:


predictions= dtree.predict(X_test)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix


# In[30]:


print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


rfc=RandomForestClassifier(n_estimators=200)


# In[33]:


rfc.fit(X_train, y_train)


# In[34]:


rfc_pred=rfc.predict(X_test)


# In[36]:


print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


# In[37]:


df['Kyphosis'].value_counts()


# In[ ]:




