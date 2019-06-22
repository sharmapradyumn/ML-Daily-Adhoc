#!/usr/bin/env python
# coding: utf-8

# # supervised machine learning

# In[2]:


import numpy as np #for array 
import pandas as pd # for data loading
from sklearn.tree  import DecisionTreeClassifier # only decision tree classifier for classification


# In[9]:


# here we do the classificatin between apple and orange
# here apple is smooth and orange is bumpy
# 0 for smooth and  1 for bumpy
features = [[100,0],[120,0],[130,1],[150,1]]  # these are features of  aplle and orange first is wait and second is smooth or bumpy


# In[10]:


labels = ["apple","apple","orange","orange"] # so this the answer for each feature that we teach the machine throgh DecisionTreeClassifier algo


# In[12]:


dtc=DecisionTreeClassifier() # calling  the DecisionTreeClassifier algo


# In[14]:


trained = dtc.fit(features,labels) # fit the data in algo and traine the machine for prediction


# In[53]:


trained.predict([[127,0],[125,1]])  # prediction of labels


# In[ ]:





# In[ ]:




