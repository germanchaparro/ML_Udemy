# Polynomial Regression

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

#%% Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#%% Predict with the linear regression
lr_y_pred = linear_regressor.predict(X)

#%% Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=2)
X_poly = poly_regressor.fit_transform(X)
linear_poly_regressor = LinearRegression()
linear_poly_regressor.fit(X_poly, y)

#%% Predict with the polynomial regression
pr_y_pred = linear_poly_regressor.predict(X_poly)

#%% Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lr_y_pred, color = 'blue')
plt.title('Salary vs Level (Linear regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#%% Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, pr_y_pred, color = 'green')
plt.title('Salary vs Level (Polynomial regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#%% Visualising the Polynomial Regression results (for higher resolution and smoother curve)
poly_regressor_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_regressor_3.fit_transform(X)
linear_poly_regressor_3 = LinearRegression()
linear_poly_regressor_3.fit(X_poly_3, y)
pr_3_y_pred = linear_poly_regressor_3.predict(X_poly_3)
plt.scatter(X, y, color='red')
plt.plot(X, pr_3_y_pred, color = 'orange')
plt.title('Salary vs Level (Polynomial regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#%% Choose the target level
target_level = 6.5
#%% Predicting a new result with Linear Regression
print(linear_regressor.predict([[target_level]]))

#%% Predicting a new result with Polynomial Regression
print(linear_poly_regressor.predict(poly_regressor.fit_transform([[target_level]])))
print(linear_poly_regressor_3.predict(poly_regressor_3.fit_transform([[target_level]])))

