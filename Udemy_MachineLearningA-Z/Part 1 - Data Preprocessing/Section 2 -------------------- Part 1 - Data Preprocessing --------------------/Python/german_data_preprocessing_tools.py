# Data Preprocessing Tools

#%% Importing the libraries
import pandas as pd                  # library to import datasets and preprocess them
import numpy as np                   # library to work with arrays
import matplotlib.pyplot as plt      # library to plot charts

#%% Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values     # get all the rows and all the columns except the last one
y = dataset.iloc[:, -1].values      # get all the rows and the last column
print(X)
print(y)

#%% Taking care of missing data
from sklearn.impute import SimpleImputer    # use SimpleImputer class to replace nan values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # we create an instance of the class saying that missing values are numpy nan, and we will replace them with the mean of each column
imputer.fit(X[:, 1:3])                      # choose numeric columns in the feature matrix
X[:, 1:3] = imputer.transform(X[:, 1:3])    # apply the mean with the transform method to the selected columns and assign them to the same columns
print(X)

#%% Encoding categorical data

#%% Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')   # transformer use the enconder OneHotEncoder, and we specify column 0 (Country) to apply the enconding. remainder set to passthrough is to avoid deleting the rest of the columns
X = np.array(ct.fit_transform(X))          # ColumnTransformer has a fit and transform method (just one step), and the result must be casted to numpy array
print(X)

#%% Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                  # use the LabelEncoder class
y = le.fit_transform(y)              # apply the fit and transform method to the y vector
print(y)

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)  # a good way to split train and test is 80% train and 20% test. The random_state = 1 is for fixing the seed and avoid different splits be made in each run (this is for learning purposes)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[:, 3:] = ss.fit_transform(X_train[:, 3:])      # fit and transform the training set using the StandardScaler class (using standardisation method, not normalization which only has to be used when the features follow a normal distribution)
X_test[:, 3:] = ss.transform(X_test[:, 3:])            # only transform the testing set (without fiting) since we have to use the fit values used for training
print(X_train)
print(X_test)
