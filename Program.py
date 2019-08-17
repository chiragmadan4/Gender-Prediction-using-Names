# -*- coding: utf-8 -*-
"""

@author: Chirag Madan
"""
import numpy as np
import pandas as pd
import re

english_check = re.compile(r'[a-z]')

# pre-labelled names
maleDf = pd.read_csv('Indian-Male-Names.csv')
femaleDf = pd.read_csv('Indian-Female-Names.csv')

df = maleDf.append(femaleDf)


# Extracting features from the name
def features(name):
    name = name.lower()
    return {
        'firstChar':name[0],
        'lastChar':name[-1],
        'isLastCharVowel': 1 if name[-1] in ['a','e','i','o','u'] else 0,
        'lastTwoChar': name[-2:]
    }

# dataset contains features for all names
dataset = pd.DataFrame(columns=['firstChar'
                                , 'lastChar'
                                , 'isLastCharVowel'
                                , 'lastTwoChar'
                                , 'gender'])

    
# Filling the dataset
idx = 0
for i,row in df.iterrows():
    name = row[0]
    
    if type(name)!=str:
        continue
    
    name = name.lower()
    firstName = name.split()[0]
    
    gender = 1          # Male-1 , Female-0
    if row[1] == 'f':
        gender = 0
    feature = features(firstName)
    
    firstChar = feature['firstChar']
    
    lastChar = feature['lastChar']
    
    lastTwoChar = feature['lastTwoChar']
    
    if not re.compile('[a-z]').match(firstChar) or not re.compile('[a-z]').match(lastChar):
        continue
    isLastCharVowel = feature['isLastCharVowel']
    
    row = [firstChar, lastChar, isLastCharVowel, lastTwoChar, gender]
    
    dataset.loc[idx] = row
    idx = idx + 1

#dataset.to_csv('Gender-predictor-features.csv', header = True)

# seperating dependent and independent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.astype('int')

# encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])

labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])

labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
X = X[: ,1:]

onehotencoder = OneHotEncoder(categorical_features=[261])
X = onehotencoder.fit_transform(X).toarray()
X = X[: ,1:]

onehotencoder = OneHotEncoder(categorical_features=[284])
X = onehotencoder.fit_transform(X).toarray()
X = X[: ,1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# Using Logistic regression model
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred_Logistic = regressor.predict(X_test)
print(regressor.score(X_test, y_test))  # accuracy achieved = 0.86015


# Using DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred_DecisionTree = classifier.predict(X_test)

print(classifier.score(X_test, y_test))  # accuracy achieved = 0.8804
