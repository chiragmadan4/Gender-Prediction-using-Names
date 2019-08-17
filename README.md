# Gender-Prediction-using-Names
Gender Prediction using Logistic Regression and Decision Tree Classifier

Gender Prediction was done using supervised learning techniques.<br>
The dataset was imported from kaggle that contained around 30k Indian names.<br>
The dataset was preprocessed and features were extracted from the names. Some eg of features extracted from names are:<br>  firstChar, lastChar, lastTwoChar, isLastCharVowel
<br>For eg: For name 'chirag': firstChar : 'c' <br>
                           lastChar  : 'g' <br>
                           lastTwoChar : 'ag' <br>
                           isLastCharVowel : 0 <br>
Feature matrix was label encoded ,one hot encoded and models were trained on it. <br>
For Logistic Regression model accuracy achieved was: 86.01% <br>
For Decision Tree Classifier accuracy achieved was: 88.04% <br>



