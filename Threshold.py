#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import datetime
from statsmodels import robust
import matplotlib as mpl
from matplotlib import gridspec
import pylab
from matplotlib import cm
from pytz import timezone
import Analysis
from scipy.stats import norm


# Adjust code so that if value is above 100, switch to just saying the value is 100

# In[1]:


def Threshold(a):
    img = cv2.imread(a,0)

    #plt.hist(img.ravel(),64,[0,256]); plt.ylim(0,1000); plt.show()

    list = img.ravel()

    def Gaussian(x,x0,sig):
        y= (1/(np.sqrt(2*np.pi)*sig))*np.exp(-(x-x0)**2/2/sig**2)
        return(y)
    #plt.hist(list,bins = 64,density=True)
    xlist = np.arange(0,250,0.01)
    ylist = Gaussian(xlist,np.mean(list),np.std(list))
    #plt.plot(xlist,ylist)
    value = np.mean(list)#-(np.std(list))
    #print("The threshold value of this image is:",value*2)
    if value>100:
        value = 100
        return(value*2)
    else:
        return(value*2)


# In[ ]:




