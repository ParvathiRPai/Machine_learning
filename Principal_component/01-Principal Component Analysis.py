#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Principal Component Analysis
# 
# Let's discuss PCA! Since this isn't exactly a full machine learning algorithm, but instead an unsupervised learning algorithm, we will just have a lecture on this topic, but no full machine learning project (although we will walk through the cancer set with PCA).
# 
# ## PCA Review
# 
# Make sure to watch the video lecture and theory presentation for a full overview of PCA! 
# Remember that PCA is just a transformation of your data and attempts to find out what features explain the most variance in your data. For example:

# <img src='PCA.png' />

# ## Libraries

# In[21]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Let's work with the cancer data set again since it had so many features.

# In[22]:


from sklearn.datasets import load_breast_cancer


# In[23]:


cancer = load_breast_cancer()


# In[24]:


cancer.keys()


# In[25]:


print(cancer['DESCR'])


# In[26]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[27]:


df.head()


# ## PCA Visualization
# 
# As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

# In[30]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()
scaler.fit(df)


# In[33]:


scaled_data = scaler.transform(df)


# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come with SciKit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().
# 
# We can also specify how many components we want to keep when creating the PCA object.

# In[34]:


from sklearn.decomposition import PCA


# In[35]:


pca = PCA(n_components=2)


# In[36]:


pca.fit(scaled_data)


# Now we can transform this data to its first 2 principal components.

# In[37]:


x_pca = pca.transform(scaled_data)


# In[38]:


scaled_data.shape


# In[39]:


x_pca.shape


# Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

# In[52]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# Clearly by using these two components we can easily separate these two classes.
# 
# ## Interpreting the components 
# 
# Unfortunately, with this great power of dimensionality reduction, comes the cost of being able to easily understand what these components represent.
# 
# The components correspond to combinations of the original features, the components themselves are stored as an attribute of the fitted PCA object:

# In[55]:


pca.components_


# In this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

# In[56]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[57]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# This heatmap and the color bar basically represent the correlation between the various feature and the principal component itself.
# 
# ## Conclusion
# 
# Hopefully this information is useful to you when dealing with high dimensional data!

# # Great Job!
