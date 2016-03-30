# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as pl
%matplotlib inline

df_raw = pd.read_csv("auto-mpg.csv")
df.default.value_counts()
df_raw[pd.isnull(df_raw).any(axis=1)]

df_raw['horsepower'].plot(kind='hist', alpha=0.5)
df_raw

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    
df = df_raw
df.horsepower[[32,126,330,336,354,374]] = df_raw.horsepower.median()


