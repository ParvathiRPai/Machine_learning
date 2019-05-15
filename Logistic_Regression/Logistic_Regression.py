
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train=pd.read_csv('titanic_train.csv')


# In[4]:


train.head()


# In[6]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[7]:


sns.set_style('whitegrid')


# In[12]:


sns.countplot(x='Survived', data=train, hue='Pclass')


# In[14]:


sns.distplot(train['Age'].dropna(), kde=False, bins=30)


# In[15]:


train['Age'].plot.hist(bins=35)


# In[16]:


train.info()


# In[17]:


sns.countplot(x='SibSp', data=train)


# In[21]:


train['Fare'].hist(bins=40, figsize=(10,4))


# In[23]:


sns.boxplot(x="Pclass", y='Age', data=train)


# In[27]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
            return Age
    


# In[29]:


train['Age']=train[['Age','Pclass']].apply(impute_age, axis=1)


# In[35]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[31]:


train.drop('Cabin', axis=1, inplace=True)


# In[32]:


train.head()


# In[34]:


train.dropna(inplace=True)


# In[37]:


sex=pd.get_dummies(train['Sex'], drop_first=True)


# In[38]:


embark=pd.get_dummies(train['Embarked'], drop_first=True)


# In[39]:


train=pd.concat([train,sex, embark], axis=1)


# In[40]:


train.head(2)


# In[47]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)


# In[48]:


train.head()


# In[49]:


train.drop(['PassengerId'], axis=1, inplace=True)


# In[50]:


train.head()


# In[52]:


X=train.drop('Survived', axis=1)
y=train['Survived']


# In[54]:


from sklearn.cross_validation import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


logmodel=LogisticRegression()


# In[61]:


logmodel.fit(X_train, y_train)


# In[62]:


predictions=logmodel.predict(X_test)


# In[63]:


from sklearn.metrics import classification_report


# In[64]:


print(classification_report(y_test, predictions))


# In[65]:


from sklearn.metrics import confusion_matrix


# In[66]:


confusion_matrix(y_test, predictions)

