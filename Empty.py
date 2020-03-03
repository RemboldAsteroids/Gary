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


# In[ ]:


def empty(a):
    # Load angdist/pixel document [ratio, xloc, yloc]
    star_ratios = np.loadtxt('loc_and_ratios.txt')

    # Find center of data
    midpoint = (720/2,480/2)

    # Loop of angles to rotate through
    rads = np.arange(0,2*np.pi,np.deg2rad(5))

    empty = []
    alt_atmosphere = 114.0

    for i in np.arange(0,len(rads)):
        # rotate original by ang radians
        ang = rads[i]
    
        # Make everything centered at (0,0)
        xs = star_ratios[:,1]-midpoint[0]
        ys = star_ratios[:,2]- midpoint[1]
    
        # Rotation matrix!
        rota = [[np.cos(ang), -np.sin(ang)],[np.sin(ang),np.cos(ang)]]
        mul = np.matmul(rota,[xs,ys])
        trans = np.transpose(mul)
    
        # Move things back to way they were
        xs2 = trans[:,0]+midpoint[0]
        ys2 = trans[:,1]+midpoint[1]
    
        # Append to array
        rotated_group = np.column_stack((star_ratios[:,0],xs2,ys2))
        if i == 0:
            all_rotated_copies = rotated_group
        else:
            all_rotated_copies = np.vstack((all_rotated_copies,rotated_group))
        
    # Only care about data points that fall within our camera pixel regions

    cond1 = (all_rotated_copies[:,1]>=0)&(all_rotated_copies[:,1]<=720)
    cond2 = (all_rotated_copies[:,2]>=0)&(all_rotated_copies[:,2]<=480)
    cond12 = cond1&cond2
    
    # Here we make the different squares have different ratio values
    
    # This inc can be changed to make squares different sized
    inc = 40
    xdivs = np.arange(0,720+inc,inc)
    ydivs = np.arange(0,480+inc,inc)
    ydivs = ydivs[::-1]

    # Create empty array
    empty = np.zeros((len(ydivs)-1,len(xdivs)-1))

    # Look in each square region
    for i in np.arange(0,len(xdivs)-1):
        left = xdivs[i]
        right = xdivs[i+1]
        for j in np.arange(0,len(ydivs)-1):
            bottom = ydivs[j+1]
            top = ydivs[j]
        
            # Look at all data points within x and y bounds
            cond_x = ((all_rotated_copies[:,1] >= left)&(all_rotated_copies[:,1] < right))
            cond_y = ((all_rotated_copies[:,2] >= bottom)&(all_rotated_copies[:,2] < top))
            cond_both = (cond_x&cond_y)
            subset = all_rotated_copies[cond_both]
        
            # If anything is in there, average the ratios and append to empty array
            if len(subset) > 0:
                empty[j,i] = np.average(subset[:,0])
            else:
                empty[j,i] = 0
    return(empty)

