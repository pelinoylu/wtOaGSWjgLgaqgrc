#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas, numpy, matplotlib ve seaborn kütüphanelerini yükleyelim
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[170]:


# Reading the data
tdm = pd.read_csv("term-deposit-marketing-2020.csv")
print(tdm.head())

# Checking column names of the data
print("\nColumn names:", tdm.columns)

# Number of rows and columns
print("\nNumber of rows and columns:",tdm.shape)


# In[3]:


#Checking whether there is a null value or not
tdm.isnull().sum()
#There is no nully value in the data


# In[4]:


# Checking NAs.
tdm.isna().any()
#There is no 'not a number' value in the data


# In[5]:


#Checking structure of the data
print(tdm.info())


# In[6]:


#Creating a summary table which includes min, max ,mean, median and standard deviation of the numeric variables
tdm.pivot_table(values=('age', 'balance', 'day', 'duration', 'campaign'
                         ), index="y", aggfunc=[min, max, np.mean, 'median', np.std], margins=True)


# In[129]:


#Distribution of job types over y ( has the client subscribed to a term deposit)
plt.figure(dpi=50)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="job",palette="magma",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 4500))


# In[147]:


plt.figure(dpi=5)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="marital",palette="mako",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 15000))


# In[148]:


plt.figure(dpi=10)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="education",palette="Spectral",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 15000))


# In[138]:


plt.figure(dpi=15)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="housing",palette="magma",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 20000))


# In[150]:


plt.figure(dpi=10)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="loan",palette="mako",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 15000))


# In[149]:


plt.figure(dpi=10)
with sns.axes_style("white"):
    g = sns.catplot(x="y", data=tdm, kind="count", hue="contact",palette="Spectral",aspect=2, height=5)
    g.set_xticklabels(step=1,fontsize=13, rotation=90)
    g.set_yticklabels(fontsize=13)
    g.ax.set(ylim=(0, 10000))


# In[169]:


#Frequencies of the categorical variable
print(pd.crosstab(index=tdm['job'], columns='job'),"\n ","\n ",
pd.crosstab(index=tdm['marital'], columns='marital'),"\n ","\n ",
pd.crosstab(index=tdm['education'], columns='education'),"\n ","\n ",
pd.crosstab(index=tdm['contact'], columns='contact'))


# In[168]:


#Frequencies of the binary variables
print(pd.crosstab(index=tdm['housing'], columns='month')
      pd.crosstab(index=tdm['loan'], columns='month'),"\n ","\n ",
      pd.crosstab(index=tdm['default'], columns='default'))


# In[13]:


#Arranging binary variables as numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
default_encoded = le.fit_transform(tdm['default'])
housing_encoded = le.fit_transform(tdm['housing'])
loan_encoded = le.fit_transform(tdm['loan'])

y_encoded = le.fit_transform(tdm['y'])


tdm['default'] = default_encoded
tdm['housing'] = housing_encoded
tdm['loan'] = loan_encoded
tdm['y'] = y_encoded

tdm.head()


# In[14]:


#Checking levels of categorical variables 
print(tdm.job.unique())
print(tdm.marital.unique())
print(tdm.education.unique())
print(tdm.contact.unique())
print(tdm.month.unique())


# In[21]:


#Turning categorical variables as dummy variables with one hot encodeding 
new_data = pd.get_dummies(tdm, columns = ['job', 'marital','education','contact','month'])
new_data.info()


# In[154]:


#Checking correlation of the variables 
corr=new_data.corr()
corr.style.background_gradient()


# In[29]:


#Pairplot of the data
#Checking the distribution of the variable
sns.pairplot(data=tdm, height=2.5,kind="reg", hue="y")


# # Modelling

# In[22]:


#Preparing the data for modelling
X = new_data.drop(columns=["y"]).values 
y= new_data["y"].values
Y= new_data["y"].values


# ### 1. Logistic Regression

# In[23]:


# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# prepare the cross-validation procedure
cv = KFold(n_splits=5, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
logistic=mean(scores)


# ### 2. Decision Tree

# In[24]:


from sklearn.tree import DecisionTreeClassifier

# Construct model
dt = DecisionTreeClassifier(max_depth=20).fit(X=X, y=y)

# Test model
y_pred = dt.predict(X=X)
# evaluate model
scores = cross_val_score(dt, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
decisiontree=mean(scores)


# ### 3. Bagging Classifier Method

# In[26]:


from sklearn.ensemble import BaggingClassifier

# Construct model
dt = DecisionTreeClassifier(max_depth=20)
bag = BaggingClassifier(dt, n_estimators=200, max_samples=0.9, random_state=42)


bagging=bag.fit(X=X, y=y)


# evaluate model
scores = cross_val_score(bagging, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
baggingc=mean(scores)


# ### 4. Random Forest

# In[28]:


from sklearn.ensemble import RandomForestClassifier

# Construct model
rf = RandomForestClassifier(n_estimators=200, max_samples=0.9, random_state=42)


randomf= rf.fit(X=X, y=y)


y_pred = randomf.predict(X=X)
# evaluate model
scores = cross_val_score(randomf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
randomforest=mean(scores)


# ### 5. AdaBoost Classifier 

# In[29]:


from sklearn.ensemble import AdaBoostClassifier

adc = AdaBoostClassifier(n_estimators=10)
ada=adc.fit(X=X, y=y)

scores = cross_val_score(ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
adaboost=mean(scores)


# ### 6. Gradient Boosting Classifier

# In[30]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.03)
gbc=gbc.fit(X=X, y=y)
# evaluate model
scores = cross_val_score(gbc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
Gradient=mean(scores)


# ### 7. Stochastic Gradient Boost

# In[31]:


sgbt = GradientBoostingClassifier(subsample=0.9, max_features=.2, n_estimators=30)
sgbt= sgbt.fit(X=X, y=y)
# evaluate model
scores = cross_val_score(sgbt, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
StochasticGradient=mean(scores)


# ### 8. XG Boost

# In[35]:


import xgboost as xgb 
xg_cl = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

xg= xg_cl.fit(X=X, y=y)
# evaluate model
scores = cross_val_score(xg, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
xgboost=mean(scores)


# # Model Comparison Part

# In[40]:


print(StochasticGradient,xgboost,Gradient,adaboost,randomforest,logistic,baggingc,decisiontree)
max(StochasticGradient,xgboost,Gradient,adaboost,randomforest,logistic,baggingc,decisiontree)


# In[92]:


Dict = {"Stochastic Gradient": round(StochasticGradient,4), "Xgboost": round(xgboost,4),
        "Gradient Boosting" : round(Gradient,4),
        "Adaboost":round(adaboost,4),"Random Forest":round(randomforest,4),"Logistic":round(logistic,4),
        "Bagging Classifier":round(baggingc,4),
        "Decision Tree":round(decisiontree,4)}
print(Dict)

data_ = list(Dict.items())
performance = np.array(data_)

print(performance)

#We can see that the xgboost has the highest accuracy level


# # CONCLUSIONS
# 
# ### Data was examined the whether there is null or na value, checking structure.
# ### It was created a summary table and checked distribution of the variables.
# ### Categoric and binary variables were arranged as dummy variables. 
# ### After arranging the variables Logistic Regression, Decision Tree, Bagging Classifier, Random Forrest, 
# ### AdaBoost Classifier, Gradient Boosting Classifier, Stochastic Gradient Boost and Xgboost machine learning algorithms were conducted.
# ### After fitting ML models were calculated accuracy levels by evaluating with 5-fold cross validation.
# ### At end of the model performance comparison, it was concluded that XgBoost is the best model to predict situation of customer will subscribe (yes/no) to a term deposit (variable y)
