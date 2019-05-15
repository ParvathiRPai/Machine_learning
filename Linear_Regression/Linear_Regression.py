#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[58]:


USAhousing.head()


# In[59]:


USAhousing.info()


# In[60]:


USAhousing.columns


# In[61]:


sns.pairplot(USAhousing)


# In[62]:


sns.distplot(USAhousing['Price'])


# In[8]:


sns.pairplot(df)


# In[10]:


sns.distplot(df['Price'])


# In[63]:


sns.heatmap(USAhousing.corr())


# In[64]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[67]:


from sklearn.linear_model import LinearRegression


# In[68]:


lm = LinearRegression()


# In[69]:


lm.fit(X_train,y_train)


# In[70]:


print(lm.intercept_)


# In[71]:


lm.coef_


# In[72]:


X_train.columns


# In[74]:


pd.DataFrame(lm.coef_,X.columns, columns=['Coeff'])


# In[76]:


from sklearn.datasets import load_boston


# In[78]:


boston=load_boston()


# In[79]:


boston.keys()


# In[80]:


print(boston['DESCR'])


# In[81]:


predictions=lm.predict(X_test)


# In[82]:


predictions


# In[83]:


plt.scatter(y_test, predictions)


# In[84]:


sns.distplot((y_test-predictions))


# In[85]:


from sklearn import metrics


# In[86]:


metrics.mean_absolute_error(y_test, predictions)


# In[87]:


metrics.mean_squared_error(y_test, predictions)


# In[88]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




