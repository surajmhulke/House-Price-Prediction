 

# House Price Prediction using Machine Learning

Have you ever experienced the challenges of buying a new house? Dealing with scams, price negotiations, and researching local areas can be quite a hassle. This project aims to address these issues by developing a machine learning model for predicting house prices based on various features.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction
In the quest to buy a house, this project offers a machine learning solution to predict house prices based on a dataset of 13 features. These features include information like the type of dwelling, lot size, configuration, and more.

The project will employ various machine learning techniques to build and evaluate models for predicting house prices. The goal is to provide accurate and reliable predictions to assist homebuyers in making informed decisions.

## Importing Libraries
In this step, we import the necessary Python libraries for data manipulation, visualization, and machine learning.
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Importing Dataset

We load the house price prediction dataset, which can be downloaded from this link.

 

dataset = pd.read_excel("HousePricePrediction.xlsx")

# Data Preprocessing

The dataset contains different types of columns, including integer, float, and object. We identify and categorize these features.

 

# Categorical variables
object_cols = [...]

# Integer variables
num_cols = [...]

# Float variables
fl_cols = [...]

# Exploratory Data Analysis (EDA)

EDA is crucial for understanding the data and discovering patterns. We use visualizations to explore relationships and distributions within the dataset.

 

# Create a correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)

Data Cleaning

Data cleaning involves handling missing values, removing irrelevant columns, and ensuring the data is in good shape for modeling.

 

# Drop the 'Id' column
dataset.drop(['Id'], axis=1, inplace=True)

# Fill missing values in 'SalePrice'
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Remove records with null values
new_dataset = dataset.dropna()

Feature Engineering

We apply one-hot encoding to categorical features to convert them into a suitable format for machine learning models.

 

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

Model Development and Evaluation

We train and evaluate regression models, including Support Vector Machine (SVM), Random Forest Regressor, and Linear Regression.

 

# Model training and evaluation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

Model and Accuracy
As we have to train the model to determine the continuous values, so we will be using these regression models.

SVM-Support Vector Machine
Random Forest Regressor
Linear Regressor
And To calculate loss we will be using the mean_absolute_percentage_error module. It can easily be imported by using sklearn library. The formula for Mean Absolute Error : 


 

SVM – Support vector Machine
SVM can be used for both regression and classification model. It finds the hyperplane in the n-dimensional plane. To read more about svm refer this.

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
 
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))
Output : 

0.18705129
Random Forest Regression
Random Forest is an ensemble technique that uses multiple of decision trees and can be used for both regression and classification tasks. To read more about random forests refer this.

from sklearn.ensemble import RandomForestRegressor
 
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
 
mean_absolute_percentage_error(Y_valid, Y_pred)
Output : 

0.1929469
Linear Regression
Linear Regression predicts the final output-dependent value based on the given independent features. Like, here we have to predict SalePrice depending on features like MSSubClass, YearBuilt, BldgType, Exterior1st etc. To read more about Linear Regression refer this.

from sklearn.linear_model import LinearRegression
 
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))
Output : 

0.187416838
CatBoost Classifier
CatBoost is a machine learning algorithm implemented by Yandex and is open-source. It is simple to interface with deep learning frameworks such as Apple’s Core ML and Google’s TensorFlow. Performance, ease-of-use, and robustness are the main advantages of the CatBoost library. To read more about CatBoost refer this.


# This code is contributed by @amartajisce
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor()
cb_model.fit(X_train, y_train)
preds = cb_model.predict(X_valid) 
 
cb_r2_score=r2_score(Y_valid, preds)
cb_r2_score
0.893643437976127

# Evaluate models using mean_absolute_percentage_error

# Conclusion

SVM has shown the best accuracy for predicting house prices, with a mean absolute percentage error of approximately 0.18. Further improvements can be achieved by exploring ensemble learning techniques like Bagging and Boosting.

By using this machine learning model, homebuyers can make more informed decisions when purchasing a new house.
