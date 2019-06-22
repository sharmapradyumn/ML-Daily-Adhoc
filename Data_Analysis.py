#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn import datasets
import time


# In[8]:


#for i in dir(datasets):
  #  print(i)
    #time.sleep(2)     # this is online datsets for use


# In[9]:


[i for i in dir(datasets) if 'load' in i]  # this  dataset  is offline  provided by  SCI-kit learn


# In[10]:


[i for i in dir(datasets) if 'cancer' in i]


# In[11]:


[i for i in dir(datasets) if 'iris' in i]


# In[ ]:




