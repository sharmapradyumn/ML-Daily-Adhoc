#!/usr/bin/env python
# coding: utf-8

# In[46]:


from sklearn.datasets import load_iris   


# In[47]:


from sklearn.datasets import load_breast_cancer


# In[48]:


from sklearn import datasets


# In[49]:


[i for i in dir(datasets) if 'load' in i]


# In[50]:


can_data=load_breast_cancer()


# In[51]:


dir(can_data)


# In[52]:


can_data.target_names


# In[53]:


can_data.feature_names


# In[54]:


can_feature=can_data.data


# In[ ]:





# In[55]:


can_label=can_data.target


# In[56]:


iris_data = load_iris()


# In[57]:


dir(iris_data)


# In[58]:


iris_data.feature_names


# In[59]:


iris_data.target_names


# In[60]:


# data with attributes
iris_features=iris_data.data


# In[61]:


# extracting labels as per features
iris_label=iris_data.target


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


plt.xlabel("length")
plt.ylabel("width")
plt.scatter(iris_features[0:,0],iris_features[0:,1],label='sepal')
plt.scatter(iris_features[0:,2],iris_features[0:,3],label='petal')


# In[64]:


#plt.xlabel("")
#plt.ylabel("")
#plt.scatter(can_feature[0:,0],can_feature[0:,1],label='
            ')
#plt.scatter(can_feature[0:,2],iris_feature[0:,3],label='petal')


# In[ ]:





# ## graph

# In[71]:


# seperate data into training and testing
from sklearn.model_selection import train_test_split


# In[72]:


train_data,test_data,train_label,test_label=train_test_split(iris_features,iris_label,test_size=0.2)


# In[73]:


from sklearn.neighbors import KNeighborsClassifier  # importing clf


# In[ ]:





# In[74]:


kclf=KNeighborsClassifier(n_neighbors=5)  # it bydefault value is 5


# In[75]:


# now applying training data
ktrained=kclf.fit(train_data,train_label)


# In[79]:


predict_output=ktrained.predict(test_data)


# In[80]:


predict_output


# In[77]:



test_label


# In[78]:


from sklearn.metrics import accuracy_score


# In[81]:


ascore=accuracy_score(predict_output,test_label)


# In[82]:


ascore


# In[83]:


#   calling  decision  tree clf
from   sklearn.tree  import  DecisionTreeClassifier
dclf=DecisionTreeClassifier()


# In[84]:


#  training  decisition  tree
dtrained=dclf.fit(train_data,train_label)


# In[85]:


#  now predicting 
dprediction=dtrained.predict(test_data)


# In[86]:


#  now  accuracy  fo  decision 
dacr=accuracy_score(test_label,dprediction)


# In[87]:


dacr


# In[ ]:


# graph between accuracy score of KNN and DTC

