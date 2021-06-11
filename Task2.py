#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# ## Data Science and Business Analytica Internship
# ## Task 2 - Prediction using Unsupervised ML
# ## Author: Yash Pahuja
# ### Problem Statement: From the ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# 

# In[5]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[18]:


df = pd.read_csv("iris.csv")
print("Data Imported Successfully")
df.head()


# In[19]:


df.info()


# In[22]:


# Adjusting the dataset
features = df.iloc[:,[0,1,2,3]].values
features


# #### Finding the optimum number of clusters for k-means classification

# In[30]:


from sklearn.cluster import KMeans
wcss = [] #Within Cluster Sum of Square

for k in range(1,15):
    kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, 
                    n_init = 10, random_state = 0 )
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    


# #### Plotting the results onto a Line graph

# In[56]:


plt.figure(figsize=(20,10), dpi=300)
plt.plot(range(1,15),wcss,"-o")
plt.title("The Elbow Method", fontsize = 28)
plt.xlabel("Number of Clusters", fontsize = 26)
plt.ylabel("WCSS",fontsize = 26)
plt.grid(True)
plt.show()


# In[60]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300,
               n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(features)
y_kmeans


# #### Visualizing the clusters

# In[74]:


plt.scatter(features[y_kmeans == 0,0], features[y_kmeans == 0, 1],
           s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(features[y_kmeans == 1,0], features[y_kmeans == 1, 1],
           s = 100, c = 'skyblue', label = 'Iris-versicolour')
plt.scatter(features[y_kmeans == 2,0], features[y_kmeans == 2, 1],
           s = 100, c = 'lightgreen', label = 'Iris-virginica')

# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1]
            , s= 100, c = 'black', label = 'Centroids')
plt.legend()

