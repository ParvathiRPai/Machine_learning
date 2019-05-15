#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import make_blobs


# In[4]:


data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


# In[7]:


plt.scatter(data[0][:,0], data[0][:,1], c=data[1])


# In[6]:


data[1]


# In[8]:


from sklearn.cluster import KMeans


# In[9]:


kmeans=KMeans(n_clusters=4)


# In[10]:


kmeans.fit(data[0])


# In[11]:


kmeans.cluster_centers_


# In[12]:


kmeans.labels_


# In[13]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[ ]:




