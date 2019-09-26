# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:00:53 2019

@author: kaspe
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
labels = digits.target


# show data
plt.figure(0)
plt.imshow(np.array(digits.data[4]).reshape(8,8), cmap='gray')

plt.figure(1)
plt.imshow(np.array(digits.data[9]).reshape(8,8), cmap='gray')

pca = PCA(n_components=2)
data_2_dim = pca.fit(data.T)

components = data_2_dim.components_

plt.figure(2)
plt.scatter(data_2_dim.components_[0,:], data_2_dim.components_[1,:], c=labels, cmap='viridis')
plt.show()