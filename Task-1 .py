#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation 

# ### Task 1 - Prediction using Supervised ML

# ### Predicting the percentage of an student based on number of study hours

# #### step 1: Importing libraries

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")
df.head()


# In[10]:


df.info()


# ### Visualizing the Data Set

# In[14]:


df.plot(x='Hours',y='Scores', style = 'o')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# ### Preparing the Data

# In[21]:


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# ### Splitting data into training and testing sets

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)


# ### Training the Data Set

# In[28]:


from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)
print("Training Completed")


# In[35]:


print("Intercept = ",l_reg.intercept_)
print("Coefficient = ",l_reg.coef_)


# ### plotting the Regression Line

# In[48]:


line = l_reg.coef_*X+l_reg.intercept_

#plotting for the test data
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# ### Making Predictions

# In[57]:


print("Hours = \n",X_test)
y_pred = l_reg.predict(X_test)
print("Predicted Scores = \n",y_pred)


# ### Comparing Actual and Predicted Scores

# In[60]:


df_comp = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
df_comp


# #### As per given task, we have to predict score of student if he/she studied for 9.25 hrs/day

# In[67]:


hours = 9.25
pred = l_reg.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# ### Evaluating using Mean Absolute Error

# In[71]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[73]:


print("Mean Squared Error: ", metrics.mean_squared_error(y_test,y_pred))


# In[74]:


print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test,y_pred))


# In[78]:


print("r2_score: ", r2_score(y_test,y_pred))

