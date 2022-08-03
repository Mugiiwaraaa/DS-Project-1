#!/usr/bin/env python
# coding: utf-8

# ## Real Estate - Price Prediction ML Model

# In[1]:


import pandas as pd


# In[2]:


housing= pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


#FOR PLOTTING HISTOGRAMS
# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# ## Training and Testing Splitting of Datasets
# 

# In[8]:


import numpy as np


# In[9]:


#For Learning purpose
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]
    


# In[10]:


# train_set,test_set=split_train_test(housing,0.2)


# In[11]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in training set {len(train_set)}\nRows in testing set {len(test_set)}")


# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[13]:


strat_train_set['CHAS'].value_counts()


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


housing=strat_train_set.copy()


# ## Looking for correlations

# In[16]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[17]:


# from pandas.plotting import scatter_matrix
# attributes = ['MEDV','RM','ZN','LSTAT']
# scatter_matrix(housing[attributes],figsize=(12,8))


# In[18]:


# housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[19]:


# housing.plot(kind="scatter",x="LSTAT",y="MEDV",alpha=0.8)


# ## Trying out new attributes

# In[20]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[21]:


housing.head(2)


# In[22]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


# housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[24]:


#Splitting the features and labels
housing=strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[25]:


#We are placing the median of RM in the missing dataplaces
median = housing['RM'].median()


# In[26]:


housing['RM'].fillna(median)


# In[27]:


housing.shape


# In[28]:


from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy="median")
imputer.fit(housing)


# In[29]:


imputer.statistics_


# In[30]:


#Now we create a transformed Dataframe that consists the imputed values
X= imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[31]:


housing_tr.describe()


# ## Creating Workflow Pipeline

# In[32]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),
                     ('std_Scaler',StandardScaler()),
                     ])
#Can add many functions to this pipeline


# In[33]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)
#This applies all the pipeline functions to the imputed np array


# In[34]:


housing_num_tr
#This is what we use as input for predictors


# ## Selecting a desired model for the problem

# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[36]:


#Now we prepare some test data and test the accuracy of the model
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]


# In[37]:


prepared_data=my_pipeline.transform(some_data)


# In[38]:


model.predict(prepared_data)


# In[39]:


list(some_labels)


# ## Evaluating the model

# In[40]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[41]:


lin_mse


# ## Using better Evaluation techniques - Cross Validation

# In[42]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[43]:


rmse_scores


# In[44]:


#defining a function to print scores, mean and std
def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[45]:


print_scores(rmse_scores)


# ## Saving the Model

# In[46]:


from joblib import dump, load
dump(model,'RealEstatePricePredictor.joblib')


# ## Testing the model on test data

# In[47]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)


# In[48]:


final_rmse


# In[ ]:




