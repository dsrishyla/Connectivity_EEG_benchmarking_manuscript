#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import xarray as xr
import numpy as np
from statsmodels.stats.anova import AnovaRM 
from scipy import stats
from matplotlib import pyplot as plt
import pingouin as pg
import numpy as np
import seaborn as sns
import scipy


# In[ ]:


subject_files = glob.glob(f'/work/srishyla/{method}/*_EC.nc')

#intra

for file in subject_files:
    var_list = 
    xarray = xr.open_dataarray(file)
    
    for sample in range(0,100):
        std = xarray.sel(bootstrap_samples=sample, region1=vn, region2=vn).values.std()
        var = std ** 2
        if var > 1:
            print(file)
            print(var)


# In[ ]:




