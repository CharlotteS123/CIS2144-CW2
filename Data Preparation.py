"""
Created on Mon Oct 25 22:08:55 2021

@author: charl
"""
# Set up libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import Datasets
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("features.csv")
df3 = pd.read_csv("stores.csv")

# Merge Datasets
df = pd.merge(df1, df2, on = ("Store", "Date", "IsHoliday"))
df = pd.merge(df, df3, on = ("Store"))
pd.set_option('display.max_columns', None)
print(df.head(5))

df.to_csv('Merged Data.csv')

# Remove Outliers
num = ["Fuel_Price", "Temperature", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", 'Size']
for i in num:
    Q1 = df[i].quantile(0.10)
    Q3 = df[i].quantile(0.90)
    IQR = Q3-Q1
    Lower = Q1 - (1.5*IQR)
    Upper = Q3 + (1.5*IQR)
    df = df[~((df[i]<Lower)|(df[i]>Upper))]

# Fill Missing Values
print(df.isna().sum())
df = df.fillna(0)

df['Day'] = df['Date'].str[0:2]
df['Month'] = df['Date'].str[3:5]
df['Year'] = df['Date'].str[6:10]
df.drop('Date', inplace=True, axis = 1)

# Convert Weekly_Sales to Categories
df["Weekly_Sales_Category"] = ""
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(-10000, 0), 'A', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(0, 5000), 'B', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(5000, 10000), 'C', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(10000, 15000), 'D', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(15000, 20000), 'E', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(20000, 25000), 'F', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(25000, 30000), 'G', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(30000, 35000), 'H', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(35000, 40000), 'I', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(40000, 45000), 'J', df['Weekly_Sales_Category'])
df['Weekly_Sales_Category'] = np.where(df['Weekly_Sales'].between(45000, 50000), 'K', df['Weekly_Sales_Category'])
df.drop('Weekly_Sales', inplace=True, axis = 1)

# Convert to Numerical
print(df.dtypes)

#df = pd.get_dummies(df, columns=["IsHoliday", "Type", "Weekly_Sales_Category], drop_first=False)
l1 = LabelEncoder()
df['IsHoliday'] = l1.fit_transform(df['IsHoliday'])
df['Type'] = l1.fit_transform(df['Type'])
df['Weekly_Sales_Category'] = l1.fit_transform(df['Weekly_Sales_Category'])


# Convert to Float
df = df.astype(float)
print(df.dtypes)

df.to_csv('Prepared Data.csv')

