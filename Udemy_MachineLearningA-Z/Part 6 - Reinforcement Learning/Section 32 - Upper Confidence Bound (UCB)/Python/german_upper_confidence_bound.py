# Upper Confidence Bound (UCB)

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#%% Implementing UCB
import math

n_users = 10000           # number of users/rounds the experiment was executed
n_ads = 10                # number of ads
ads_selected = []                 # full list of ads selected over the round n. Will grow up to n_users


numbers_of_selections = [0] * n_ads
sums_of_rewards = [0] * n_ads
total_reward = 0

confidence_interval = np.zeros((n_ads,), dtype=float)

for n in range(0, n_users):
    selected_ad = 0
    max_upper_bound = 0
    for i in range(0, n_ads):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i  = math.sqrt( ( 3/2 ) * ( math.log(n+1) / numbers_of_selections[i] ) )
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            selected_ad = i
    ads_selected.append(selected_ad)
    
    numbers_of_selections[selected_ad] += 1
    
    reward = dataset.values[n, selected_ad]
    sums_of_rewards[selected_ad] += reward
    total_reward = total_reward + reward

#%% Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of adds selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()