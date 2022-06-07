#!/usr/bin/env python
# coding: utf-8

# # Task 1
# ## Prediction using Supervised Machine Learning
# #### Author: Abin Johnson
# #### Submitted to : The Sparks Foundation

# ### Importing and assesing dataset 

# In[4]:


##Importing important libraries---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


link =  "http://bit.ly/w-data"
set1 = pd.read_csv(link)
print("Data is successfully imported")
set1


# In[4]:


# To get percentiles,mean,std,max,count of the given dataset, let's use describe method

set1.describe()


# In[5]:


set1.info()


# ### Visualization of the data

# In[6]:


import seaborn as sns
plt.boxplot(set1)
plt.show()


# In[7]:


## Analyzing the data with Scatter plot
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.title('Hours studied vs Score', fontsize=15)
plt.scatter(set1.Hours,set1.Scores,color='blue',marker='*')
plt.show()


# ##### Analysis of Scatterplot: As we can see in this Scatterplot, Scores and Hours are POSITIVELY RELATED. This implies that if a student studies more hours, more marks will be attained by the students.

# In[10]:


# We can use iloc function to retrieve a particular value belonging to a row and column using the index values assigned to it.
# So let's see each coordinates.

X = set1.iloc[:,:-1].values
X
Y = set1.iloc[:,1].values
Y
print("X coordinates are:", X,"Y Coordinates are:",Y)


# ### Training, Testing and Splitting of the Dataset

# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state =0, test_size = 0.2 )
# We splitted our data using 80-20 rule.(test_size = 0.2)

print("X trained data shape = ",X_train.shape)
print("X test data shape =",X_test.shape)
print("Y train data shape = ",Y_train.shape)
print("Y test data shape = ",Y_test.shape)


# ### Linear Regression of Training Data Set

# In[12]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

# Let's train our Algorithm
lr.fit(X_train,Y_train)
print("B0 =",lr.intercept_,"\nB1 =",lr.coef_)
## B0 is Intercept & Slope of the line is B1.,"


# In[14]:


# Plotting the regression line
Y0 = lr.intercept_ + lr.coef_*X_train
plt.scatter(X_train,Y_train,color='red',marker='*')
plt.plot(X_train,Y0,color='green')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line (Train set)",fontsize=15)
plt.grid()
plt.show()


# ### Linear Regression Analysis of Test Data

# In[15]:


Y_pred=lr.predict(X_test)##predicting the Scores for test data
print(Y_pred)


# In[16]:


Y_test


# In[17]:


#plotting line on test data
plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='green',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=15)
plt.grid()
plt.show()


# ### Comparison of Actual and Predicted Values

# In[18]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# ### Finding the Accuracy (Goodness of Fit)

# In[19]:


from sklearn import metrics
metrics.r2_score(Y_test,Y_pred) 
# Goodness of Fit


# ##### This shows that our model is 94% accurate, that is its a best fitted model

# ### Predicting the Error

# In[20]:


MSE = metrics.mean_squared_error(Y_test,Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error      = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error     = ",Abs_E)


# ### Predicting the Scores

# In[21]:


Prediction_score = lr.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)


# ### Inference

# ##### Question: What will be the predicted score if a student studies for 9.25/hrs a day?
# ##### As shown above, we can conclude that if a student studies for 9.25hrs a day, he may secure approximately 93.69% marks.
