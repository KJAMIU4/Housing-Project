#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load the data
housing_data = pd.read_csv("C:\\Desktop\\PROJECT\\Housing.csv")
housing_data


# In[3]:


housing_data.info()


# In[4]:


from sklearn.model_selection import train_test_split


X = housing_data.drop(["price"],axis = 1)
y = housing_data["price"]


# In[5]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[6]:


train_data = X_train.join(y_train)
train_data


# In[7]:


train_data.hist(figsize=(12,8))


# In[8]:


train_data.furnishingstatus.value_counts()


# In[9]:


train_data = train_data.join(pd.get_dummies(train_data.furnishingstatus)).drop("furnishingstatus",axis=1)


# In[10]:


# Replace specific strings with boolean values
train_data["updated_basement"] = train_data['basement'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_prefarea"] = train_data['prefarea'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_mainroad"] = train_data['mainroad'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_guestroom"] = train_data['guestroom'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_hotwaterheating"] = train_data['hotwaterheating'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_airconditioning"] = train_data['airconditioning'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})

# Convert the column to boolean type
train_data["updated_basement"] = train_data["updated_basement"].astype(bool)
train_data["updated_prefarea"] = train_data["updated_prefarea"].astype(bool)
train_data["updated_mainroad"] = train_data["updated_mainroad"].astype(bool)
train_data["updated_guestroom"] = train_data["updated_guestroom"].astype(bool)
train_data["updated_hotwaterheating"] = train_data["updated_hotwaterheating"].astype(bool)
train_data["updated_airconditioning"] = train_data["updated_airconditioning"].astype(bool)

# Drop the former basement column

train_data = train_data.drop(columns=["basement","prefarea","mainroad","guestroom","hotwaterheating","airconditioning"])
train_data


# In[11]:


plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")


# In[112]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train= train_data.drop(["price"],axis = 1)
y_train = train_data["price"]
X_train_s = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train,y_train)


# In[13]:


test_data = X_test.join(y_test)

test_data = test_data.join(pd.get_dummies(test_data.furnishingstatus)).drop("furnishingstatus",axis=1)

# Replace specific strings with boolean values
test_data["updated_basement"] = test_data['basement'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
test_data["updated_prefarea"] = test_data['prefarea'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
test_data["updated_mainroad"] = test_data['mainroad'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
test_data["updated_guestroom"] = test_data['guestroom'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
test_data["updated_hotwaterheating"] = test_data['hotwaterheating'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
test_data["updated_airconditioning"] = test_data['airconditioning'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})

# Convert the column to boolean type
test_data["updated_basement"] = test_data["updated_basement"].astype(bool)
test_data["updated_prefarea"] = test_data["updated_prefarea"].astype(bool)
test_data["updated_mainroad"] = test_data["updated_mainroad"].astype(bool)
test_data["updated_guestroom"] = test_data["updated_guestroom"].astype(bool)
test_data["updated_hotwaterheating"] = test_data["updated_hotwaterheating"].astype(bool)
test_data["updated_airconditioning"] = test_data["updated_airconditioning"].astype(bool)

# Drop the former basement column

test_data = test_data.drop(columns=["basement","prefarea","mainroad","guestroom","hotwaterheating","airconditioning"])
test_data


# In[222]:


X_test= test_data.drop(["price"],axis = 1)
y_test = test_data["price"]
X_test_s = scaler.fit_transform(X_test)

model.score(X_test,y_test)


# In[224]:


from sklearn.ensemble import RandomForestRegressor


forest = RandomForestRegressor()
forest.fit(X_train,y_train)


# In[226]:


forest.score(X_test,y_test)


# In[228]:


from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid ={
    "n_estimators": [6,8,10,20],
    "max_features":[2,4,6,8]
}

grid_search = GridSearchCV(forest,param_grid,cv=5,scoring="neg_mean_squared_error",
                            return_train_score=True)
grid_search.fit(X_train,y_train)


# In[229]:


best_forest = grid_search.best_estimator_


# In[230]:


best_forest.score(X_test,y_test)


# In[126]:


#rom sklearn.svm import SVR


#support = SVR(kernel='rbf') 
#support.fit(X_train,y_train)


# In[128]:


#support.score(X_test,y_test)


# In[ ]:




