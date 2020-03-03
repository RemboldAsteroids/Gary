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


# In[5]:


def SkyArea(a,b,c,d,e): 
    '''
    #Defines the function that will be used to analyse a single image captured by the D6 
    all-Sky camera. a is the image, b is the height of the kernel, c is the width of the kernel, 
    d is the value at which each pixel will be thrshholded to the value of e, ussualy e should be 
    255 to get a black and white image.
    '''
    Cloudmap = cv2.imread(a,0) #Reads in the captured image as  CV2 image that has a list of values for each pixel in the image
    kernel = np.ones((b,c),np.uint8) #determines the area that will be changed for each part of the picture

    #cv2.imshow(a,Cloudmap) #Shows the image before it's converted into a black and white, threshed, image
    #cv2.waitKey(0)
    #cv2.destroyAllWindows

    ret,threshCloud = cv2.threshold(Cloudmap,d,e,cv2.THRESH_BINARY_INV) #switches all pixels to either a 0 value of the value of 3 given in the function at the top. 
    threshCloudOpen = cv2.morphologyEx(threshCloud, cv2.MORPH_OPEN, kernel) #"opens" the pixels, cleaning up any extraneous pixels that are surrounded by a different value.
    threshCloudFinal = cv2.morphologyEx(threshCloudOpen, cv2.MORPH_CLOSE, kernel) #"closes" the pixel values, alligning the pixels so that the picture looks cleaned up
    #cv2.imshow("Threshcloud", threshCloudFinal) #shows the finished threshold image, for comparison to the original image.
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    
        
    #pic = cv2.imread(a,0)
        
    # Threshold it at d
    #ret,th3 = cv2.threshold(pic,d,e,cv2.THRESH_BINARY)
    
    # Use b and c for kernal in morpohology close and erode
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(b,c))
    
    # Close up any small gaps in clouds and then open back up other areas
    #closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=1)
    #opening = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=1)
    
        # Load angdist/pixel document [ratio, xloc, yloc]
    star_ratios = np.loadtxt('loc_and_ratios.txt')

    # Find center of data
    midpoint = (720/2,480/2)

    # Loop of angles to rotate through
    rads = np.arange(0,2*np.pi,np.deg2rad(5))

    empty = []

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
    
    # Initialize sum of angular distance covered by clouds
    alt_atmosphere = 114.0
    sum_angdist = 0
    for j in np.arange(0,len(threshCloudFinal)):
        for k in np.arange(0,len(threshCloudFinal[0])):
            if (threshCloudFinal[j,k] != 0):
                box_x = divmod(k,40)[0]
                box_y = divmod(j,40)[0]
                cont = empty[box_y,box_x]
            
                 # Use formulas to calcualte rectangular solid angle area
                cont_rad = np.deg2rad(cont)
                cont_solid_ang = 4*np.arcsin(np.sin(cont_rad/2)**2)
                contribution = cont_solid_ang*alt_atmosphere**2
                sum_angdist += contribution
    return(sum_angdist)
    


# In[ ]:




