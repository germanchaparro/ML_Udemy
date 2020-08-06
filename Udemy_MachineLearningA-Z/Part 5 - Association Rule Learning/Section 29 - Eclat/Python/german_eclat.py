# Eclat

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
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



# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))