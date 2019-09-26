# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:35:56 2019

@author: Morten Sahlertz
"""

from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names


#%% New variable names
median_income = X[:, np.newaxis, 0]
median_house_value = y


#%% Histogram for median income
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.hist(median_income, bins=100) # histogram
ax.set_title('Median Income Histogram')
ax.set_xlabel('Median Income')
ax.set_ylabel('Occurences')


#%% Standard deviation, mean and median
standard = np.std(median_income)
print('Spredningen er: {0:.5f}'.format(standard))
mean = np.mean(median_income)
print('Middelværdien er: {0:.5f}'.format(mean))
median = np.median(median_income)
print('Medianen er: {0:.5f}'.format(median))


#%% Histogram for median income with fitted standard deviation
xarr = np.linspace(np.min(median_income), np.max(median_income), 500)
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.set_title('Median Income Histogram Fitted with Standard Deviation')
ax.set_xlabel('Median Income')
ax.plot(xarr, norm.pdf(xarr, mean, standard)) # prob. dens. func.


#%% Correlation between median_income and median_house_value
correlation = np.correlate(np.array(median_income[:,0]), median_house_value, "full")

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.plot(correlation)
ax.set_title('Median Income and Median House Value Cross Correlation')
ax.set_xlabel('Samples')
ax.set_ylabel('Correlation Value')

corrcoef = np.corrcoef(median_income[:,0], y)
print('Korrelations Koefficient er: {0:.5f}'.format(corrcoef[1,0]))


#%% Percentile calculations
percentile_5 = np.percentile(y, 5)
print('Den 5 percentile er: {}$'.format(int(percentile_5*100000))
percentile_95 = np.percentile(y, 95)
print('Den 95 percentile er: {}$'.format(int(percentile_95*100000))


#%% Histogram plot for median house value
fig4 = plt.figure()
xarr2 = np.linspace(np.min(y), np.max(y), 500)
ax1 = fig4.add_subplot(211)
ax1.hist(y, bins=100)
ax1.set_title('Median House Value Histogram')
ax1.set_xlabel('Median House Value')
ax1.set_ylabel('Occurences')
ax2 = fig4.add_subplot(212)
ax2.plot(xarr2, norm.pdf(xarr2, np.mean(y), np.std(y))) # prob. dens. func.
ax2.set_title('Median House Value Histogram Fitted with Standard Deviation')
ax2.set_xlabel('Median House Value')
plt.tight_layout()
plt.show()