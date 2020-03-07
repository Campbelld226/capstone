#!/usr/bin/env python
# coding: utf-8

# # Visualizations

# In[1]:


import pandas as pd
from Labeller import Labeller


# In[2]:



l = Labeller(pd.read_csv('test/CoRoT-2b_1.7.txt'))
df = l.get_cluster()


# # Countplot of unique values

# In[3]:


display(df['labels_desc'].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
sns.countplot(x='labels_desc',data = df)
plt.title('Count of Unique Periods')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


import matplotlib.pyplot as plt 
for col in df:
    if df[col].dtype != (object):
        sns.distplot(df[col])
        plt.show()
        sns.scatterplot(x=range(len(df[col])), y=df[col])
        plt.show()


# In[ ]:




