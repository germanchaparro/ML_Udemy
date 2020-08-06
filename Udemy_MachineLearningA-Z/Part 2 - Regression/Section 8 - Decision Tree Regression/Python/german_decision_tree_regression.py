# Decision Tree Regression

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

#%% Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)                               # when working with trees or forest there is no need to do feature scaling

#%% Predicting a new result
prediction = regressor.predict([[6.5]])

#%% Visualising the Decision Tree Regression results (higher resolution)
step = 0.1
X_grid = np.arange(min(X), max(X) + step, step)    # changing to add step to avoid removing the last value of the array
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary vs Level (Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()