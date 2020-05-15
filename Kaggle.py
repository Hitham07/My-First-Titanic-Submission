#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math, time, random, datetime
import numpy as np
import pandas as pd
#---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import missingno
#---------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
#---------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import *
#---------------------------------------------------------


# In[2]:


train = pd.read_csv("C:\\Users\\HithamPC\\Desktop\\Train.csv")
test = pd.read_csv("C:\\Users\\HithamPC\\Desktop\\test.csv")


# In[3]:


train.dtypes


# In[4]:


train.isnull().sum()


# In[5]:


train.count()


# In[6]:


train.info()


# In[7]:


missingno.matrix(train)


# In[8]:


train.Age.plot.hist()


# In[9]:


train.Sex[train.Sex == 'male'] = 1
train.Sex[train.Sex == 'female'] = 0


# In[10]:


train=train.fillna(train.mean())


# In[13]:


train.isnull().sum()


# In[12]:



train = train.dropna(subset=['Embarked'])


# In[ ]:





# In[14]:



train=train.drop("Ticket", axis=1)
train=train.drop("Cabin", axis=1)


# In[ ]:





# In[15]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[16]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[17]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[18]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[21]:


train=train.drop(["PassengerId","Name"], axis=1)


# In[22]:


train.head()


# In[23]:


#train=train.drop(["PassengerId","Name","Title"],axis=1)

one_hot_cols = train.columns.tolist()
one_hot_cols.remove("Age")
one_hot_cols.remove("Survived")
one_hot_cols.remove("Fare")


trained = pd.get_dummies(train, columns=one_hot_cols)

trained.head()


# In[24]:



# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# In[25]:


X_train = trained.drop('Survived', axis=1) # data
y_train = trained.Survived # labels


# In[26]:


X_train.isnull().sum()


# In[27]:


y_train.shape


# In[28]:


# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               y_train, 
                                                                    30)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# In[29]:


# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  y_train, 
                                                  10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))


# In[30]:


start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 
                                                                      X_train, 
                                                                      y_train, 
                                                                           10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))


# In[31]:


# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))


# In[32]:


# Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))


# In[33]:


# Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 
                                                                       X_train, 
                                                                       y_train,
                                                                       10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))


# In[ ]:


X_train.head()


# In[34]:


# Define the categorical features for the CatBoost model
cat_features = np.where(X_train.dtypes != np.float)[0]
cat_features


# In[35]:


# Use the CatBoost Pool() function to pool together the training data and categorical feature labels
train_pool = Pool(X_train, 
                  y_train,
                  cat_features)


# In[36]:


# CatBoost model definition
catboost_model = CatBoostClassifier(iterations=1000,
                                    custom_loss=['Accuracy'],
                                    loss_function='Logloss')

# Fit CatBoost model
catboost_model.fit(train_pool,
                   plot=False)

# CatBoost accuracy
acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)


# In[37]:


# Print out the CatBoost model metrics
print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))


# In[38]:


trained.head(2)


# In[52]:


test.head(2)
#test.isnull().sum()


# In[69]:


tested = test.dropna(subset=['Embarked'])
tested=tested.drop(["PassengerId","Name","Cabin","Ticket"] , axis=1)
tested["Age"]=test.fillna(tested["Age"].mean())
tested.Sex[tested.Sex == 'male'] = 1
tested.Sex[tested.Sex == 'female'] = 0




one_hot_cols = tested.columns.tolist()
one_hot_cols.remove("Age")
one_hot_cols.remove("Fare")

testedx = pd.get_dummies(tested, columns=one_hot_cols)


# In[71]:


testedx.head()
testedx=testedx.drop("Parch_9" , axis=1)


# In[66]:





# In[74]:


predictions =  catboost_model.predict(data=testedx)

submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions # our model predictions on the test dataset
submission.head(100)


# In[75]:


submission.dtypes


# In[76]:


if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")


# In[77]:


# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('C:\\Users\\HithamPC\\Desktop\\catboost_submission.csv', index=False)
print('Submission CSV is ready!')


# In[78]:


import xgboost


# In[79]:


pip list


# In[80]:


import xgboost as xgb


# In[ ]:




