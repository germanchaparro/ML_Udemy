# Artificial Neural Network for Regression

# Importing the libraries

#%% Part 1 - Data Preprocessing
import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)

#%% Importing the dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)  # a good way to split train and test is 80% train and 20% test. The random_state = 1 is for fixing the seed and avoid different splits be made in each run (this is for learning purposes)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()                    # For ANN we need to scale all of the features
X_train = ss.fit_transform(X_train)      # fit and transform the training set using the StandardScaler class (using standardisation method, not normalization which only has to be used when the features follow a normal distribution)
X_test = ss.transform(X_test)            # only transform the testing set (without fiting) since we have to use the fit values used for training
# print(X_train)
# print(X_test)

#%% Part 2 - Building the ANN

#%% Initializing the ANN
ann = tf.keras.models.Sequential()            # create an ann

#%% Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # creating a fully connected (Dense class) hidden layer with 6 neurons and each neuron using rectifier activation function

#%% Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # creating a second fully connected (Dense class) hidden layer with 6 neurons and each neuron using rectifier activation function

#%% Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))  # creating the output layer with 1 neuron. No activation function required since it is Regression.

#%% Part 3 - Training the ANN

#%% Compiling the ANN
# adam optimizer is for stochastic gradient descent
# loss is mean squared error
ann.compile(optimizer='adam', loss='mean_squared_error')

#%% Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)     # training a stochastic ann with batch of 32 (recommended value) and 100 epochs (always choose a "big" number of epochs)

#%% Predicting the results of the test data
y_pred = ann.predict(X_test)

print (y_pred)

#%% Compare real values vs predicted values
np.set_printoptions(precision=4)
print( np.concatenate( (y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1) ), 1 ) )
