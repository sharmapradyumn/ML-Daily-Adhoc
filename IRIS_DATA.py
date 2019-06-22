#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris  # this is iris plant data 


# In[3]:


import matplotlib.pyplot as plt #for  graph ploting


# In[4]:


from sklearn.tree import DecisionTreeClassifier  # Decision algo for decision making


# In[7]:


from sklearn.metrics import accuracy_score    # check the accuracy score of machine and algo


# In[8]:


iris = load_iris()


# In[9]:


dir(iris)  # inside iris data file


# In[10]:


iris.feature_names  # these are the features 


# In[11]:


iris.target_names  # these are the labels or answers


# In[17]:


label=iris.target  # 0--> cetosa 1--> versicolor 2---> virginica
label.shape


# In[16]:


features=iris.data  # this is the  data of features
features.shape


# In[14]:


iris.filename # simple filemane


# In[15]:


iris.DESCR   # descripton about data


# In[19]:


SL=features[0:,0]
SW=features[0:,1]


# In[20]:


plt.xlabel("length")
plt.ylabel("width")
plt.scatter(SL,SW,label="sepal_data",marker='*')
plt.scatter(features[0:,2],features[0:,3],label="petal_data",marker='x')
plt.legend()


# In[21]:


#  now time  for seperating  data   into two category  
# 1 . --training  data
#  2. --testing  data -- Questions 
from   sklearn.model_selection   import  train_test_split
train_data,test_data,label_train,label_test=train_test_split(features,label,test_size=0.1)


# In[25]:


#calling  decisiontree classifier  
clf=DecisionTreeClassifier()
#  now time for  training  clf 
trained=clf.fit(train_data,label_train)  ##################################3


# In[26]:


#  now predicting  flowers
predicted_flowers=trained.predict(test_data)

predicted_flowers  #  algo answer




# In[27]:


label_test#  actual answer


# In[28]:


#  find  accuracy  score  
accuracy_score(label_test,predicted_flowers)


# In[ ]:




