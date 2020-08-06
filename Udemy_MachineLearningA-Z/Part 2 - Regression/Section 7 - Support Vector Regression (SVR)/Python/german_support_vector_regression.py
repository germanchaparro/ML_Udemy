# Support Vector Regression (SVR)

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

y = y.reshape(len(y), 1)   # we have to reshape it to a 2D array due to the StandardScaler clase that will perform the feature scaling expects a 2D array as input

print(y)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()         # use different scalers for X and y since when executing the fit function it uses the mean and standard deviation of the features that are being transformed.
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

print(X)
print(y)

#%% Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')     # create an SVR with rbf kernel
regressor.fit(X, y)


#%% Predicting a new result
level_position = [[6.5]]                                        # the input value must be a 2D array
level_position = sc_X.transform(level_position)                    # have to transform to the same scale the input level using sc_X
scaled_prediction = regressor.predict(level_position)           # execute the prediction, but the output value is scaled in terms of sc_y
real_prediction = sc_y.inverse_transform(scaled_prediction)  # reverse the scale using the inverse_transform function of sc_y


#%% Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Salary vs Level (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()                                 # salary for level position is treated as an outlier in SVR, due to it prediction for position 10 is not well predicted

#%% Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Salary vs Level (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
