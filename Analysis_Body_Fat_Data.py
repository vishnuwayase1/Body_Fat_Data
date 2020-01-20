#!/usr/bin/env python
# coding: utf-8

# In[18]:


f=open("F:\DATA\dss\Ml_datasets/BodyFatdata.csv")
head=f.readline()
lines=f.readlines()
print(lines)


# In[19]:


import pandas as pd


# In[20]:


df=pd.read_csv("F:\DATA\dss\Ml_datasets/BodyFatdata.csv")
#df.head()
print(df)


# In[21]:


ints=df.iloc[:,0:4]
#print(ints)
y=df.iloc[:,4:5]
print(y)


# In[22]:


import numpy as np
x=np.array(ints)
print(x)


# In[23]:


Y=np.array(y)
print(Y)


# In[24]:


one=np.ones((len(df),1))
X=np.concatenate((one,x),axis=1)
print(len(X))


# In[25]:


[X.shape[0]*0.7]


# In[26]:


int(X.shape[0]*0.7)
int(Y.shape[0]*0.7)


# In[27]:


xtrain=X[:int(X.shape[0]*0.7),:]
xtest=X[int(X.shape[0]*0.7):,:]
print(len(xtrain))
print(len(xtest))


# In[28]:


ytrain=Y[:int(X.shape[0]*0.7),:]
print(ytrain.shape)
ytest=Y[int(X.shape[0]*0.7):,:]
print(ytest.shape)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ycap=model.predict(xtest)
from sklearn import metrics
metrics.accuracy_score(ytest,ycap)
#T=np.c_[ytest,ycap]


# In[30]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)
ycap1=model.predict(xtest)
T1=np.c_[ytest,ycap1]
metrics.accuracy_score(ytest,ycap1)


# In[31]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)
ycap2=model.predict(xtest)
T2=np.c_[ytest,ycap2]
metrics.accuracy_score(ytest,ycap2)


# In[32]:


from sklearn.svm import SVC
model=SVC()
model.fit(xtrain,ytrain)
ycap3=model.predict(xtest)
metrics.accuracy_score(ytest,ycap3)


# In[33]:


def train(model):
    model.fit(xtrain,ytrain)
    ycap=model.predict(xtest)
    acc=metrics.accuracy_score(ytest,ycap)
    return [model,acc]
    


# In[34]:


train(model)


# In[ ]:





# In[ ]:





# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


plt.figure(figsize=(13,9))


# In[52]:


sns.boxplot(xtrain)


# In[53]:


sns.boxplot(xtest)


# In[56]:


sns.boxplot(ytrain)


# In[55]:


sns.boxplot(ytest)


# In[58]:


import numpy as np


# In[72]:


plt.subplots(figsize=(13, 9))
sns.heatmap(xtrain,annot=True)


# In[89]:


plt.figure(figsize=(13,9))
sns.distplot(xtrain[::,-1])
sns.boxplot(xtrain[::,-1])


# In[ ]:





# In[85]:


plt.figure(figsize=(13,9))
sns.distplot(np.log1p(xtrain[::,-1]))


# In[103]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ycap11=model.predict(xtest)
t11=np.c_[ytest,ycap11]
metrics.accuracy_score(ytest,ycap2)
#print(t11)


# In[99]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)
ycap2=model.predict(xtest)
T2=np.c_[ytest,ycap2]
metrics.accuracy_score(ytest,ycap2)


# In[ ]:




