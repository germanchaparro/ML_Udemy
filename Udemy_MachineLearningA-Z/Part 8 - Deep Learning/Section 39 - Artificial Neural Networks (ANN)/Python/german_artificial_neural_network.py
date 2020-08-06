# Artificial Neural Network

# Importing the libraries

#%% Part 1 - Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)

#%% Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%% Encoding categorical data

#%% Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                  # use the LabelEncoder class
X[:, 2] = le.fit_transform(X[:, 2])              # apply the fit and transform method to the Gender column

#%% One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')   # transformer use the enconder OneHotEncoder, and we specify column 1 (Country) to apply the enconding. remainder set to passthrough is to avoid deleting the rest of the columns
X = np.array(ct.fit_transform(X))          # ColumnTransformer has a fit and transform method (just one step), and the result must be casted to numpy array

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)  # a good way to split train and test is 80% train and 20% test. The random_state = 1 is for fixing the seed and avoid different splits be made in each run (this is for learning purposes)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()                    # For ANN we need to scale all of the features
X_train = ss.fit_transform(X_train)      # fit and transform the training set using the StandardScaler class (using standardisation method, not normalization which only has to be used when the features follow a normal distribution)
X_test = ss.transform(X_test)            # only transform the testing set (without fiting) since we have to use the fit values used for training
print(X_train)
print(X_test)

#%% Part 2 - Building the ANN

#%% Initializing the ANN
ann = tf.keras.models.Sequential()            # create an ann

#%% Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # creating a fully connected (Dense class) hidden layer with 6 neurons and each neuron using rectifier activation function

#%% Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # creating a second fully connected (Dense class) hidden layer with 6 neurons and each neuron using rectifier activation function

#%% Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # creating the output layer with 1 neuron (since it is a binary output 0 or 1) and each neuron using sigmoid activation function

#%% Part 3 - Training the ANN

#%% Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

#%% Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)     # training a stochastic ann with batch of 32 (recommended value) and 100 epochs (always choose a "big" number of epochs)

#%% Part 4 - Making the predictions and evaluating the model


#%% Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""
new_input = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
new_input = ss.transform(new_input)
print(ann.predict(new_input))

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

#%% Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

from sklearn.metrics import accuracy_score
acc_sco = accuracy_score(y_test, y_pred)
print(acc_sco)
