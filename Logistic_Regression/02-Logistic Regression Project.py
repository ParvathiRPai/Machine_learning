
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[2]:


ad_data=pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[3]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[4]:


ad_data.info()
ad_data.describe()


# In[42]:





# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[6]:


ad_data['Age'].plot.hist(bins=30)


# **Create a jointplot showing Area Income versus Age.**

# In[7]:


sns.jointplot(x="Age",y="Area Income", data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[12]:


sns.jointplot(x='Age', y='Daily Time Spent on Site',data=ad_data, kind='kde', color='red')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[14]:


sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage',data=ad_data)


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[17]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[18]:


from sklearn.cross_validation import train_test_split


# In[25]:



X=ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
y=ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)


# In[23]:


data=ad_data
data.head()


# ** Train and fit a logistic regression model on the training set.**

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[28]:


logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[29]:


predictions=logmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# ## Great Job!
