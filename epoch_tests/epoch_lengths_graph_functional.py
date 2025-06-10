#!/usr/bin/env python
# coding: utf-8

# In[12]:


import glob
import pandas as pd
import xarray as xr
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


# In[19]:

epoch_lens = ['1','2','4','6','8','10','12','14','16','18','20']
subjects = ['sub-032312','sub-032310','sub-032304','sub-032302','sub-032307']
methods = ['coh','ciplv', 'imcoh', 'wpli2']

# In[28]:

for method in methods:
    for subject in subjects:
        means_list = []
        stds_list = []
        for length in epoch_lens:
            xarray_EC = xr.open_dataarray(f'{subject}_array_{method}_{length}_EC.nc')
            mean = np.mean(xarray_EC.values)
            std = np.std(xarray_EC.values)
            means_list.append(mean)
            stds_list.append(std)
        
        plt.scatter(epoch_lens,means_list,label=method,c='b')
        plt.errorbar(epoch_lens,means_list, yerr=stds_list,c='b')
        plt.xlabel("Epoch length (s)")
        plt.ylabel("Mean connectivity (EC)")
        plt.legend()
        plt.title("Mean differences across epoch lengths")
        plt.savefig(f'{subject}_epoch_lengths_{method}.png')


# In[29]:


# In[ ]:




