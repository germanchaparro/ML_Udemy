# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

#%% Install apyori library from the command prompt
#pip install apyori   # we have to install this library since scikitlearn doesn have Apriori modules
                      # then restart the kernel to be able to use it
                      # then check the library was successfully installed with the command pip list | grep apy

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)   # since the dataset doesn't have column titles, we must tell pandas to exclude headers while reading the dataset

# apyori requires the feature data to be formated as a string list and not as a pandas dataframe, so we heve to converrt it to a list
transactions = []
n_rows = len(dataset)
n_cols = len(dataset.columns)

# for loop to traverse n_rows and n_cols dataset. Casting the values to str since it is required by apyori
for i in range(0, n_rows):
    transactions.append([str(dataset.values[i, j]) for j in range(0, n_cols)])


#%% Training the Apriori model on the dataset
from apyori import apriori
# min_support = (3 * 7) / 7501  (the product appears minimum 3 times in the 7 days divided by the number of transactions - based specifically on the problem we are solving-) = 0.0027 ~= 0.003  
# min_confidence = 0.2 (rule of thumb after tryint 0.8, then 0.4, then 0.2)
# min_lift = 3 (rule of thumb, based on experience)
# min_length = 2, minimum number of elements you need to have in your rule. Choose 2 because the business problem states that we want to offer one product and get other for free
# max_length = 2, maximum number of elements you need to have in your rule. Choose 2 because the business problem states that we want to offer one product and get other for free

rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

#%% Visualising the results
results = list(rules)
print(results)

#%% Displaying the first results coming directly from the output of the apriori function
## Displaying the first results coming directly from the output of the apriori function
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#%% Putting the results well organised into a Pandas DataFrame
## Putting the results well organised into a Pandas DataFrame
print(resultsinDataFrame)

#%% Displaying the results non sorted
## Displaying the results non sorted
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))


## Displaying the results sorted by descending lifts
