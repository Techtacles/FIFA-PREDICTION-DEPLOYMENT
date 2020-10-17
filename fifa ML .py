#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
sns.set()


# In[2]:


fifa=pd.read_csv("data.csv")



# In[3]:


to_stay=["ID","Name","Age","Nationality","Club","Overall","Potential","Value","Wage","Real Face"]
fifa.drop(fifa.columns.difference(to_stay),axis="columns",inplace=True)


# In[4]:


fifa.head()


# In[5]:


fifa.set_index("ID",inplace=True)


# In[6]:


fifa.head()


# In[7]:


fifa.isnull().sum()


# In[8]:


fifa["Club"].dropna(axis="rows",inplace=True)


# In[9]:


fifa.dropna(axis="rows",inplace=True)


# In[10]:


fifa.isnull().sum()


# In[11]:


fifa.shape


# In[12]:


fifa.head(20)


# Checking for outliers in Age.

# In[13]:


sns.boxplot(fifa["Age"])


# In[14]:


q1=np.percentile(fifa["Age"],25)
q3=np.percentile(fifa["Age"],75)
iqr=q3-q1
lower=q1-(1.5*iqr)
upper=q3+(1.5*iqr)
fifa["Age"][(fifa["Age"]<np.abs(lower))|(fifa["Age"]>upper)].max()


# Hence, there are no outliers in Age.

# In[15]:


fifa.Value.unique


# In[16]:


#fifa.info()


# The goal here is to have only one Value column of their value in euros

# In[17]:


fifa["Value2"]=fifa["Value"].apply(lambda x: x.split("€")[1])


# In[18]:


fifa["Value3"]=fifa["Value2"].apply(lambda x:x.split("M")[0]*1000000 if x.split("M")==True else x.split("K")[0]*1000)
fifa


# In[19]:


fifa["Value3"]=fifa["Value2"].apply(lambda x: 1 if "M" in x else 0)
fifa["Value4"]=fifa["Value2"].apply(lambda x : x.split("M")[0] if "M" in x else x.split("K")[0]).astype(float)


# In[20]:


fifa["Value5"]=(fifa[fifa["Value3"]==1]["Value4"]*1000000)
a=fifa[fifa["Value3"]==0]
fifa["Value5"].fillna(a["Value4"]*1000,inplace=True)


# In[21]:


fifa


# In[22]:


fifa.drop(["Value4","Value3","Value2","Value"],axis="columns",inplace=True)


# Rename value5 column to value

# In[23]:


fifa.rename(columns={"Value5":"Value"},inplace=True)


# In[24]:


fifa[fifa["Value"]==60000]


# Change wage to value

# In[25]:


fifa["Wage"].unique()


# In[26]:


fifa["Wage2"]=fifa["Wage"].apply(lambda x: x.split("€")[1])
fifa["Wage3"]=fifa["Wage2"].apply(lambda x:x.split("K")[0]).astype(float)*1000
fifa.head()


# In[27]:


fifa.drop(["Wage","Wage2"],axis="columns",inplace=True)


# In[28]:


fifa.rename(columns={"Wage3":"Wage"},inplace=True)


# In[29]:

# In[30]:


# # MACHINE LEARNING

# Using Player features like Overall, Potential,Age to predict Player's Value and wage

# In[55]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[56]:


model=LinearRegression()


# In[57]:


#fifa.head()


# In[58]:


fifa=fifa.reset_index()


# In[59]:


X=fifa[["Age","Overall","Potential","Wage"]]


# In[60]:


y=fifa["Value"]


# Using Train test split

# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)


# In[63]:



# In[65]:


gbr=GradientBoostingRegressor().fit(Xtrain,ytrain)
#print(f"Your training accuracy is {gbr.score(Xtrain,ytrain)} while your test accuracy is {gbr.score(Xtest,ytest)}")


# In[66]:


ypred=gbr.predict(Xtest)


# In[67]:


#gbr.predict([[20,76,78,50000]])


# In[68]:


#mean_absolute_error(ytest,ypred)


# In[69]:


#mean_squared_error(ytest,ypred)


# In[70]:


#np.sqrt(mean_squared_error(ytest,ypred))


# In[72]:


pickle.dump(gbr,open("model.pkl","wb"))


# In[73]:


model=pickle.load(open("model.pkl","rb"))

