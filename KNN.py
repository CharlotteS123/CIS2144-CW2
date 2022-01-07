# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 20:11:57 2021

@author: charl
"""
# importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# preparing the data
df = pd.read_csv('Prepared Data.csv')
df1 = pd.read_csv('Prepared Test Data.csv')
df1 = df1.iloc[:,:]
df2 = pd.read_csv('Test.csv')

# splitting the data
X = df.iloc[:,:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = KNeighborsClassifier(n_neighbors=12) #n_neighbors=2
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy : ", accuracy_score(y_test,y_pred)*100)
print("Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

test_pred = classifier.predict(df1)
output = pd.DataFrame({'Store':df2.Store, 'Dept':df2.Dept, 'Date':df2.Date, 'IsHoliday':df2.IsHoliday,'Weekly_Sales_Category':test_pred})
output.to_csv('Test Results - KNN.csv')

