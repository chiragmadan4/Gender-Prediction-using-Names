# Gender-Prediction-using-Names
Gender Prediction using Logistic Regression and Decision Tree Classifier

Gender Prediction was done using supervised learning techniques.
The dataset was imported from kaggle that contained around 30k Indian names.
The dataset was preprocessed and features were extracted from the names. Some eg of features extracted from names are:  firstChar, lastChar, lastTwoChar, isLastCharVowel
for eg: For name 'chirag': firstChar : 'c'
                           lastChar  : 'g'
                           lastTwoChar : 'ag'
                           isLastCharVowel : 0
Feature matrix was label encoded ,one hot encoded and models were trained on it.
For Logistic Regression model accuracy achieved was: 86.01%
For Decision Tree Classifier accuracy achieved was: 88.04%



