# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:18:55 2019

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

food_name = "apple_pie"

img_dir = "/Users/Morten/Desktop/IHA Documents/ITMAL/food101/images/" + food_name # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
reshape_data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    reshape = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    reshape_data.append(reshape)
        
#%% Histogram for colors
fig1 = plt.figure(food_name + ' colors')
fig1.suptitle(food_name + ' colors')

ax1 = fig1.add_subplot(221)
ax1.hist(reshape[0:], bins=256) # histogram
ax1.set_title('Color Value Histogram')
ax1.set_xlabel('Color Value')
ax1.set_ylabel('Occurences')

ax2 = fig1.add_subplot(222)
ax2.hist(reshape[0:,0], bins=256, color='b')
ax2.set_title('Blue Color Value Histogram')
ax2.set_xlabel('Color Value')
ax2.set_ylabel('Occurences')

ax3 = fig1.add_subplot(223)
ax3.hist(reshape[0:,1], bins=256, color='r')
ax3.set_title('Red Color Value Histogram')
ax3.set_xlabel('Color Value')
ax3.set_ylabel('Occurences')

ax4 = fig1.add_subplot(224)
ax4.hist(reshape[0:,2], bins=256, color='g')
ax4.set_title('Green Color Values Histogram')
ax4.set_xlabel('Color Value')
ax4.set_ylabel('Occurences')

plt.tight_layout()

#%%
food_name2 = "caesar_salad"

img_dir = "/Users/Morten/Desktop/IHA Documents/ITMAL/food101/images/" + food_name2 # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
reshape_data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    reshape = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    reshape_data.append(reshape)

#%% Histogram for colors
fig2 = plt.figure(food_name2 + ' colors')
fig2.suptitle(food_name2 + ' colors')

ax1 = fig2.add_subplot(221)
ax1.hist(reshape[0:], bins=256) # histogram
ax1.set_title('Color Value Histogram')
ax1.set_xlabel('Color Value')
ax1.set_ylabel('Occurences')

ax2 = fig2.add_subplot(222)
ax2.hist(reshape[0:,0], bins=256, color='b')
ax2.set_title('Blue Color Value Histogram')
ax2.set_xlabel('Color Value')
ax2.set_ylabel('Occurences')

ax3 = fig2.add_subplot(223)
ax3.hist(reshape[0:,1], bins=256, color='r')
ax3.set_title('Red Color Value Histogram')
ax3.set_xlabel('Color Value')
ax3.set_ylabel('Occurences')

ax4 = fig2.add_subplot(224)
ax4.hist(reshape[0:,2], bins=256, color='g')
ax4.set_title('Green Color Values Histogram')
ax4.set_xlabel('Color Value')
ax4.set_ylabel('Occurences')

plt.tight_layout()