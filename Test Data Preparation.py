# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:56:37 2022

@author: charl
"""

# Set up libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import Datasets
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("features.csv")
df3 = pd.read_csv("stores.csv")

# Merge Datasets
df = pd.merge(df1, df2, on = ("Store", "Date", "IsHoliday"))
df = pd.merge(df, df3, on = ("Store"))
pd.set_option('display.max_columns', None)

# Fill Missing Values
print(df.isna().sum())
df = df.fillna(0)

df['Day'] = df['Date'].str[0:2]
df['Month'] = df['Date'].str[3:5]
df['Year'] = df['Date'].str[6:10]
df.drop('Date', inplace=True, axis = 1)

# Convert to Numerical
print(df.dtypes)

#df = pd.get_dummies(df, columns=["IsHoliday", "Type"], drop_first=False)
l1 = LabelEncoder()
df['IsHoliday'] = l1.fit_transform(df['IsHoliday'])
df['Type'] = l1.fit_transform(df['Type'])

# Convert to Float
df = df.astype(float)
print(df.dtypes)

df.to_csv('Prepared Test Data.csv')

