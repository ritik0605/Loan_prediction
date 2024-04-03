#!/usr/bin/env python
# coding: utf-8

# # Title: Loan Approval Prediction Model using Machine Learning

# # Objective:The objective of this project is to develop a machine learning model that can accurately predict whether a loan application should be approved or denied based on various applicant attributes

# # Problem Statement: The problem at hand is to create a predictive model that assists financial institutions in automating the loan approval process. By leveraging historical loan data and applicant information, the goal is to build a model that can efficiently assess the risk associated with each loan application and make reliable decisions regarding loan approval

# # Tools and Technologies:
# # Python: Programming language for data preprocessing, model training, and evaluation.
# # Libraries: pandas, scikit-learn for data manipulation, model building, and evaluation.
# # Jupyter Notebook: Environment for conducting exploratory data analysis, documenting the project, and presenting results.

# In[1]:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# # Data Exploration and Preprocessing
# In[2]:
df=pd.read_csv(r"C:\Users\Ritik Sonwane\Downloads\archive (2)\loan_sanction_train.csv")  # Load the dataset

# In[3]:
df.head(50)  # Display the first few rows

# In[4]:
df.info()  # Display column data types and missing values

# In[5]:
df.isnull().sum() #missing values in set

# In[6]:
df.describe().T

# In[7]:
df.shape

# In[8]:
df.columns


# # Feature Engineering
# In[9]:
df['loanAmount_log']= np.log(df[ 'LoanAmount'])
df['loanAmount_log'].hist(bins=20)

# In[10]:
df.isnull().sum()

# In[11]:
df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

# In[12]:
#lets fill those null values in all the respective columns
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log=df.loanAmount_log.fillna(df.loanAmount_log.mean())
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode() [0],inplace=True)
df.isnull().sum()


# # Split the Data
# In[13]:
# we will select specific columns for training and testing
x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values

# In[14]:
x
# In[15]:
y

# In[16]:
# Percentage of missing gender 
print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))

# In[17]:
# Number of people who take loan as group by gender
print("number of people who take loan as group by gender: ")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette='Set1')

# In[18]:
print("number of people who take loan as group by Marital status: ")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,palette='Set1')

# In[19]:
print("number of people who take loan as group by Dependents: ")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,palette='Set1')

# In[20]:
print("number of people who take loan as group by Self_Employed: ")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette='Set1')

# In[21]:
print("number of people who take loan as group by LoanAmount: ")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette='Set1')

# In[22]:
print("number of people who take loan as group by Credit_History: ")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette='Set1')


# # Training and testing Dataset
# In[23]:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x=LabelEncoder()


# # fit and transfer data for training data
# In[24]:
for i in range(0, 5):
    X_train[:,i]=Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]=Labelencoder_x.fit_transform(X_train[:,7])

X_train

# In[25]:
Labelencoder_y=LabelEncoder()
y_train=Labelencoder_y.fit_transform(y_train)
y_train


# # for testing data
# In[26]:
for i in range(0,5):
    X_test[:,i]= Labelencoder_x.fit_transform(X_test[:,1])
    X_test[:,7] = Labelencoder_x.fit_transform(X_test[:,7])

X_test

# In[27]:
Labelencoder_y=LabelEncoder()
y_test=Labelencoder_y.fit_transform(y_test)
y_test

# In[28]:
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
x_test=ss.fit_transform(X_test)

# # Now we are going to see which classifiers is best for accuracy
# In[29]:
from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier()
rf_clf.fit(X_train,y_train)

# In[30]:
from sklearn import metrics
y_pred = rf_clf.predict(x_test)

print("acc of random forest clf is", metrics.accuracy_score(y_pred, y_test))
y_pred

# In[31]:
from sklearn.naive_bayes import GaussianNB
nb_clf=GaussianNB()
nb_clf.fit(X_train,y_train)


# In[34]:
y_pred=nb_clf.predict(X_test)
print("acc of gaussianNB is %.", metrics.accuracy_score(y_pred,y_test))
y_pred


# In[35]:
from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[36]:
y_pred=dt_clf.predict(X_test)
print("acc of DT is", metrics.accuracy_score(y_pred,y_test))
y_pred


# In[37]:
from sklearn.neighbors import KNeighborsClassifier
kn_clf=KNeighborsClassifier()
kn_clf.fit(X_train,y_train)


# In[38]:
y_pred=kn_clf.predict(X_test)
print("acc of KN is",metrics.accuracy_score(y_pred,y_test))
y_pred
