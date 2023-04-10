#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install seaborn


# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[21]:


df=pd.read_csv("spam.csv",encoding="latin-1")


# In[22]:


# understanding the dataset
df.head(10)


# In[23]:


df.shape


# In[24]:


#to check target values are binary are not
np.unique(df["class"])


# In[25]:


np.unique(df["message"])


# In[26]:


#creating sparse matrix
x=df["message"].values
y=df["class"].values

cv=CountVectorizer()

x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[27]:


first_col=df.pop("message")
df.insert(0,'message',first_col)
df


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state=15)


# In[30]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(X_train,y_train)

y_pred_train=bnb.predict(X_train)
y_pred_test=bnb.predict(X_test)


# In[33]:


#training score
print(bnb.score(X_train,y_train)*100)

#testing score
print(bnb.score(X_test,y_test)*100)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred_train))


# In[37]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))

