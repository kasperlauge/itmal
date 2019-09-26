# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:35:56 2019

@author: User
"""

from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

median_inc = X[:, np.newaxis, 0]

#fig = plt.figure()
#fig.plot
plt.scatter(median_inc[:,0], y)
plt.xlabel('Median_Income')
plt.ylabel('Price')

mean = np.mean(median_inc)
median = np.median(median_inc)
standard = np.std(median_inc)
variance = np.var(median_inc)

#fig = plt.figure()
xarr = np.linspace(np.min(median_inc), np.max(median_inc), 500)
fig, ax = plt.subplots(2,1, figsize=(10,20)) 
ax[0].hist(median_inc, bins=100) # histogram
ax[1].plot(xarr, norm.pdf(xarr, mean, standard)) # prob. dens. func.
test = np.array(median_inc[:,0])
correlation = np.correlate(np.array(median_inc[:,0]), y, "full")
#correlation = np.correlate(median_inc, np.transpose(y))

plt.figure()
plt.plot(correlation)
plt.show()

corrcoef = np.corrcoef(median_inc[:,0], y)

percentile_5 = np.percentile(y, 5)
percentile_95 = np.percentile(y, 95)