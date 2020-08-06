# Simple Linear Regression

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Encoding categorical data
# No need to encode since there is no categorical data in this dataset

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#%% Feature Scaling
# no need to do feature scaling since there is only one feature.

#%% Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#%% Predicting the Test set results
y_pred = lr.predict(X_test)

#%% Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#%% Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#%% Making a single prediction (for example the salary of an employee with 12 years of experience)
print(lr.predict([[12]]))  # Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

#%% Getting the final linear regression equation with the values of the coefficients
print(lr.coef_)
print(lr.intercept_)            # Salary = 9312.58 × YearsExperience + 26780.09
