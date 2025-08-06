#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import ibraries
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


#visuaize data
train_data.hist(figsize=(12,8))


# In[8]:


train_data.furnishingstatus.value_counts()


# In[9]:


# Replace furnishingstatus column with boolean 
train_data = train_data.join(pd.get_dummies(train_data.furnishingstatus,drop_first=True)).drop("furnishingstatus",axis=1)


# In[10]:


# Replace specific strings with boolean values
bool_cols = ['basement', 'prefarea', 'mainroad', 'guestroom', 'hotwaterheating', 'airconditioning']
for col in bool_cols:
    train_data[f"updated_{col}"] = train_data[col].str.lower().map({'yes': 1, 'no': 0})
train_data.drop(columns=bool_cols, inplace=True)

"""
train_data["updated_basement"] = train_data['basement'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_prefarea"] = train_data['prefarea'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_mainroad"] = train_data['mainroad'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_guestroom"] = train_data['guestroom'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_hotwaterheating"] = train_data['hotwaterheating'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})
train_data["updated_airconditioning"] = train_data['airconditioning'].replace({'yes': True, 'no': False, 'YES': True, 'No': False})

# Convert the column to boolean type
train_data["updated_basement"] = train_data["updated_basement"].astype(int)
train_data["updated_prefarea"] = train_data["updated_prefarea"].astype(int)
train_data["updated_mainroad"] = train_data["updated_mainroad"].astype(int)
train_data["updated_guestroom"] = train_data["updated_guestroom"].astype(int)
train_data["updated_hotwaterheating"] = train_data["updated_hotwaterheating"].astype(int)
train_data["updated_airconditioning"] = train_data["updated_airconditioning"].astype(int)

# Drop the former basement column

train_data = train_data.drop(columns=["basement","prefarea","mainroad","guestroom","hotwaterheating","airconditioning"])
train_data
"""


# In[11]:


train_data


# In[12]:


plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")


# In[13]:


#Train and test the data using LinearRegression

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X_train= train_data.drop(["price"],axis = 1)
y_train = train_data["price"]
X_train_s = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_s,y_train)


# In[14]:


test_data = X_test.join(y_test)

# Replace furnishingstatus column with boolean 
test_data = test_data.join(pd.get_dummies(test_data.furnishingstatus,drop_first=True)).drop("furnishingstatus",axis=1)

# test_data = test_data.join(pd.get_dummies(test_data.furnishingstatus)).drop("furnishingstatus",axis=1)

# Replace specific strings with boolean values

for col in bool_cols:
    test_data[f"updated_{col}"] = test_data[col].str.lower().map({'yes': 1, 'no': 0})
test_data.drop(columns=bool_cols, inplace=True)

""" Replace specific strings with boolean values
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
"""
test_data


# In[28]:


X_test= test_data.drop(["price"],axis = 1)
y_test = test_data["price"]
X_test_s = scaler.transform(X_test)


from sklearn.metrics import root_mean_squared_error, mean_absolute_error

preds = model.predict(X_test_s)

print("R² Score:", model.score(X_test_s, y_test))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", root_mean_squared_error(y_test, preds))
#model.score(X_test_s,y_test)


# In[30]:


#Improving the model by using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


forest = RandomForestRegressor()
forest.fit(X_train_s,y_train)


# In[36]:


forest.score(X_test_s,y_test)


# In[38]:


from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid ={
    "n_estimators": [6,8,10,20],
    "max_features":[2,4,6,8]
}

grid_search = GridSearchCV(forest,param_grid,cv=5,scoring="neg_mean_squared_error",
                            return_train_score=True)
grid_search.fit(X_train_s,y_train)


# In[40]:


best_forest = grid_search.best_estimator_


# In[44]:


preds = best_forest.predict(X_test_s)
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", root_mean_squared_error(y_test, preds))

#best_forest.score(X_test,y_test)


# In[48]:


importances = best_forest.feature_importances_
features = X_train.columns  # Use this instead of X_train_s.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()



# In[ ]:




