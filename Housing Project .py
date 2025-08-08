#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[2]:


#Load the data
df = pd.read_csv("C:\\Desktop\\PROJECT\\Housing.csv")
df


# In[3]:


df.info()


# In[4]:


def preprocess_housing_data(df):
    df = df.copy()
    df = df.join(pd.get_dummies(df["furnishingstatus"]).astype(int))
    df.drop("furnishingstatus", axis=1, inplace=True)
    for col in ['basement', 'prefarea', 'mainroad', 'guestroom', 'hotwaterheating', 'airconditioning']:
        df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
    return df

updated_df = preprocess_housing_data(df)


# In[5]:


updated_df


# In[6]:


plt.figure(figsize=(12,8))
sns.heatmap(updated_df.corr(),annot=True,cmap="YlGnBu")


# In[7]:


updated_df["price_per_sqft"] = updated_df["price"] / df["area"]
updated_df["bathrooms_per_bedroom"] = updated_df["bathrooms"] / (updated_df["bedrooms"] + 1)
updated_df["bedrooms_per_area"] = updated_df["bedrooms"] / updated_df["area"]


# In[8]:


# 1. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(updated_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[9]:


updated_df


# In[10]:


X = updated_df.drop(["price"],axis = 1)
y = updated_df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scale,y_train)


# In[12]:


preds = model.predict(X_test_scale)

print("R² Score:", model.score(X_test_scale, y_test))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", root_mean_squared_error(y_test, preds))
#model.score(X_test_s,y_test)


# In[13]:


# Predicted vs Actual
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=preds, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # perfect line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual")
plt.show()


# In[14]:


#Improving the model by using RandomForestRegressor


# Tuned Random Forest
forest = RandomForestRegressor(
    n_estimators=500,        # More trees for stability
    max_depth=15,            # Prevents overfitting by limiting depth
    min_samples_split=5,     # Needs at least 5 samples to split a node
    min_samples_leaf=2,      # At least 2 samples per leaf
    max_features='sqrt',     # Random subset of features for each split
    random_state=42,         # Reproducibility
    n_jobs=-1                # Use all CPU cores
)

# Fit the model
forest.fit(X_train_scale, y_train)

# Evaluate
train_r2 = forest.score(X_train_scale, y_train)
test_r2 = forest.score(X_test_scale, y_test)

print(f"Train R²: {train_r2}")
print(f"Test R²: {test_r2}")



# In[15]:


# Feature importance
importances = forest.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.show()


# In[16]:


# Base model
rf = RandomForestRegressor(random_state=42)

# Parameter grid for tuning
param_grid = {
    'n_estimators': [200, 500, 800],         # More trees = better but slower
    'max_depth': [None, 10, 20, 30],         # None = fully grown trees
    'min_samples_split': [2, 5, 10],         # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],           # Min samples per leaf node
    'max_features': ['sqrt', 'log2']         # Features considered per split
}


# In[17]:


# Grid search with 5-fold CV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='r2'  # could also try 'neg_root_mean_squared_error'
)

grid_search.fit(X_train, y_train)


# In[18]:


# Best model after tuning
best_rf = grid_search.best_estimator_

# Evaluate
print("Best parameters:", grid_search.best_params_)
print("Train R²:", best_rf.score(X_train, y_train))
print("Test R²:", best_rf.score(X_test, y_test))


# In[19]:


importances = forest.feature_importances_
features = X_train.columns  # Use this instead of X_train_s.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()



# In[ ]:




