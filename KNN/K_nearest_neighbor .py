
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('Classified Data')


# In[5]:


df.head()


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler=StandardScaler()


# In[9]:


scaler.fit(df.drop('TARGET CLASS', axis=1))


# In[10]:


scaled_features=scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[12]:


scaled_features


# In[14]:


df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[15]:


df_feat.head()


# In[16]:


from sklearn.cross_validation import train_test_split


# In[17]:


X=df_feat
y=df["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[21]:


knn.fit(X_train,y_train)


# In[22]:


pred=knn.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix


# In[28]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[33]:


error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[34]:


plt.figure(figsize=(10,6))


# In[35]:


plt.plot(range(1,40),error_rate)


# In[39]:


knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

