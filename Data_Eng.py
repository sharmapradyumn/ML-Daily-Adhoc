#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


# reading csv file from url
df =pd.read_csv('http://13.234.66.67/summer19/datasets/info.csv')


# In[31]:


df.info()


# In[32]:


df


# In[33]:


X=df.iloc[:,0:].values
X


# In[34]:


# removing missing value or replacing with some relevant data
df.describe()


# In[35]:


from sklearn.preprocessing import Imputer


# In[36]:


imp=Imputer(missing_values='NaN',axis=0,strategy="mean")


# In[37]:


impute=imp.fit(X[:,1:3])  #  this is only fitting of columns that we want to process


# In[38]:


# time for trsnsforming the fitted columns
X[:,1:3]=impute.transform(X[:,1:3])


# In[39]:


X


# In[40]:


# string label int /float
from sklearn.preprocessing import LabelEncoder


# In[41]:


cont = LabelEncoder()  # this is country labeling


# In[42]:


# now apply column first in this LabelEn
X[:,0]=cont.fit_transform(X[:,0])
X


# In[15]:


# labelin for last column
p=LabelEncoder()


# In[16]:


X[:,3]=p.fit_transform(X[:,3])


# In[17]:


X


# In[18]:


# Now encoding first column ------making subcolumn of first column

from sklearn.preprocessing import OneHotEncoder


# In[19]:


OneHotEncoder()



# In[20]:


first_column=OneHotEncoder(categorical_features=[0])   # definig exact column number where we want to make category


# In[21]:


X=first_column.fit_transform(X).toarray()  # after transformation we need to convert into numpy array


# In[22]:


#X
X.astype(int)   # type converted into integer or to show in int only


# In[ ]:





# In[ ]:




