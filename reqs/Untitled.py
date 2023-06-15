#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


df = pd.read_csv('car data.csv')

# In[3]:


df.head()

# In[4]:


df.shape

# In[5]:


Y = df.Selling_Price

# In[6]:


df.info()

# In[7]:


cat_feature = [col for col in df.columns if df[col].nunique() < 20]

# In[8]:


cat_feature

# In[9]:


df.isnull().sum()

# In[10]:


df = df.drop_duplicates()

# In[11]:


df.duplicated().sum()

# We are dropping the Car_Name feature as it dont play any important role in Math

# In[12]:


df = df.drop(['Car_Name'] , axis = 1)  

# "no_of_year" tells how old the car is with respect to present year i.e 2023.

# In[13]:


df['no_of_year'] = 2023 - df['Year']

# In[14]:


df.drop(['Year'] , axis = 1 , inplace = True)

# In[15]:


df.head()

# Here I have seperated the categorical columns from the dataset I have also droped 'Owner' as it is already binary feature so no need to do any encoding

# In[57]:


cat_cols = [col for col in df.columns if df[col].nunique() < 5 ]
cat_df = df[cat_cols].drop(['Owner'], axis = 1)

# In[58]:


cat_df.head()

# In[62]:


from sklearn.preprocessing import OneHotEncoder
OH_Enc = OneHotEncoder(drop = 'first' , max_categories = 3)

OH_Enc.fit(cat_df)
encoded_cat_df = pd.DataFrame(OH_Enc.transform(cat_df))

# The problem will face during naming of encoded dataset since , I can type names of each columns for encoded date but in real practice it is not advisable as it can lead to human error. so I will now use inbuilt function of python for this. 

# In[96]:


encoded_df = pd.get_dummies(df , drop_first = True , dtype = 'int64')

# In[97]:


encoded_df.head()

# In[98]:


df.head()

# In[99]:


encoded_df.corr()

# In[100]:


sns.pairplot(encoded_df)

# In[102]:


corrmat=encoded_df.corr()
corr_features=corrmat.columns

plt.figure(figsize=(20,20)) 
# #plot heat map
g=sns.heatmap(corrmat,annot=True,cmap="flag")

# In[103]:


X = encoded_df.drop(['Selling_Price'], axis = 1)
y = encoded_df.Selling_Price

# In[104]:


X.head()

# In[108]:


#Feature Importance is used for Extracting importance feature
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

# In[107]:


model.feature_importances_

# In[109]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# From this section I will be working on training of model

# In[112]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 

# In[120]:


X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 69 , shuffle = True , test_size = 0.2)

# In[177]:


from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(n_estimators = 100 , random_state =69)
model1.fit(X_train , y_train)
y_pred1 = model1.predict(X_test)

# In[178]:


plt.scatter( y_test, y_pred1 )
plt.xlabel('Actual')
plt.ylabel('Predictions')

# In[180]:


mae_model1 = mean_absolute_error(y_test , y_pred1)
mae_model1

# In[124]:


X_train.shape

# In[141]:


X_test.shape

# In[143]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# In[145]:


from sklearn.model_selection import RandomizedSearchCV

# In[146]:


random_grid = {'n_estimators':n_estimators , 
               'max_features':max_features , 
              'max_depth': max_depth ,
              'min_samples_split':min_samples_split ,
              'min_samples_leaf':min_samples_leaf }

# In[162]:


rf = RandomForestRegressor()
model2 = RandomizedSearchCV( estimator = rf, param_distributions = random_grid , n_iter = 20, scoring='neg_root_mean_squared_error',cv=5,verbose=4)

# In[163]:


model2.fit(X_train , y_train)


# In[169]:


y_pred2 = model2.predict(X_test)

# In[170]:


sns.distplot(y_test-y_pred2)

# In[171]:


plt.scatter(y_test , y_pred2)
plt.xlabel('Actual')
plt.ylabel('Predictions')

# In[172]:


mae_model2 = mean_absolute_error(y_pred2 , y_test)
mae_model2 

# In[184]:


mae_model1

# As observed above hyper parameter tuning performed good compared to **RandomForestRegressor**

# In[193]:


from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
model3.fit(X_train , y_train)
y_pred3 = model3.predict(X_test)
mae_model3 = mean_absolute_error(y_pred3 , y_test)

# In[194]:


mae_model3

# In[195]:


plt.scatter(y_test , y_pred3)
plt.xlabel('Actual')
plt.ylabel('Predictions')

# In[198]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model2.pkl', 'wb')
# dump information to that file
pickle.dump(model2, file)

# In[203]:


import session_info
session_info.show()

# In[210]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



