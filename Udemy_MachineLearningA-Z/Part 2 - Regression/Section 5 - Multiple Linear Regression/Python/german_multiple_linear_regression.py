# Multiple Linear Regression

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as ply

#%% Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')   # transformer use the enconder OneHotEncoder, and we specify column 0 (Country) to apply the enconding. remainder set to passthrough is to avoid deleting the rest of the columns
X = np.array(ct.fit_transform(X))          # ColumnTransformer has a fit and transform method (just one step), and the result must be casted to numpy array
print(X)
# There is no need to remove one encoding variable to avoid the Dummy variable trap due to sklearn executes that for you


#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# There is no need to implement backward or forward elimintaion since sklearn does that for you.

#%% Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#%% Predicting the Test set results
y_pred = lr.predict(X_test)

#%% Compare real values vs predicted values
np.set_printoptions(precision=2)
print( np.concatenate( (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1) ), 1 ) )

#%% making a single prediction
print(lr.predict([[1, 0, 0, 160000, 130000, 300000]]))  #company based in california with R&D=160000, Administrative=130000 and Marketing spend=300000 

#%% printing the coefficients
print(lr.coef_)          # [ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
print(lr.intercept_)     # 42467.52924854249
