# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:50:20 2021

@author: charl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('Prepared Data.csv')
df1 = pd.read_csv('Prepared Test Data.csv')
df1 = df1.iloc[:,:]
df2 = pd.read_csv('Test.csv')

X = df.iloc[:,:-1]
y = df.iloc[:, -1]

skf = StratifiedKFold(n_splits=8)
skf.get_n_splits(X, y)

i = 1
score = 0
for train_index, test_index in skf.split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy For Split",i,": ",accuracy_score(y_test,y_pred)*100)
    score += accuracy_score(y_test, y_pred)*100
    i += 1
print("Overall Accuracy For All Splits :",(score/8))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("-------------------------------------------")
print("Accuracy : ", accuracy_score(y_test,y_pred)*100)
print("Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
test_pred = clf.predict(df1)
output = pd.DataFrame({'Store':df2.Store, 'Dept':df2.Dept, 'Date':df2.Date, 'IsHoliday':df2.IsHoliday,'Weekly_Sales_Category':test_pred})


