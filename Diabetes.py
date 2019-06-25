#!/usr/bin/env python
# coding: utf-8

# In[8]:


#from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sb


# In[9]:


df=pd.read_csv('http://13.234.66.67/summer19/datasets/diabetest.csv')


# In[11]:


# now printing schema 
df.info()


# In[12]:


#description of data
df.describe()


# In[13]:


df.head(5)


# In[15]:


# plot particular column with count
sb.countplot(df['Pregnancies'])


# In[18]:


df.hist(figsize=(15,20)) # this is for histograms


# In[20]:


sb.scatterplot(df['Pregnancies'],df['Glucose'])


# In[21]:


sb.pairplot(df)


# In[22]:


# extract attribute from dataframe
features=df.iloc[:,0:8].values


# In[24]:


# now label
label=df.iloc[:,8].values


# In[25]:


label


# In[27]:


label.shape


# In[31]:


# sep training and testing data
from sklearn.model_selection import train_test_split


# In[32]:


X,x,Y,y=train_test_split(features,label,test_size=0.2)


# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


clf=DecisionTreeClassifier()


# In[ ]:




