# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the Data Set
data = pd.read_csv("train.csv")
null = data.isnull().sum() #To determine the the no. of NaN

# Imputing the NaN Values , other methods such as kNN, sklearn Imputer can also be used
#Education
data.workclass.value_counts(sort=True)
data.workclass.fillna('Private',inplace=True)

#Occupation
data.occupation.value_counts(sort=True)
data.occupation.fillna('Prof-specialty',inplace=True)

#Native Country
data['native.country'].value_counts(sort=True)
data['native.country'].fillna('United-States',inplace=True)

# Label Encoding
from sklearn import preprocessing
for x in data.columns:
    if data[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[x].values))
        data[x] = lbl.transform(list(data[x].values))

# Creating Randomforest Model        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

y = data['target']
del data['target'] # removes the index

X = data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)

#make prediction and check model's accuracy
prediction = clf.predict(X_test)
acc =  accuracy_score(np.array(y_test),prediction)
print ('The accuracy of Random Forest is {}'.format(acc))
