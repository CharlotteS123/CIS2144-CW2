# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:52:45 2021

@author: charl
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'Type', 'CPI', 'IsHoliday', 'Unemployment', 'Size']

df = pd.read_csv("Merged Data.csv")
df = df.iloc[:,1:]

for col in columns:
      sns.pairplot(df, x_vars = col, y_vars = 'Weekly_Sales', kind = 'scatter')
      plt.show()

fig = plt.figure(figsize=(18, 14))
corr = df.corr()
c = plt.pcolor(corr)
plt.yticks(np.arange(0.5, len(corr.index), 1), corr.index)
plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
fig.colorbar(c)

sns.scatterplot( x = 'Date', y = 'Weekly_Sales', hue = 'IsHoliday', data = df)
plt.show()


