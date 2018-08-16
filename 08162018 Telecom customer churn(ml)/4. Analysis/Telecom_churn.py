# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('churn.csv')
X = dataset.iloc[:, 1:-2]
y = dataset.iloc[:, [-1]]

# Generating dummy variables for  data
X = pd.get_dummies(X,drop_first=True)
y=  pd.get_dummies(y,drop_first=True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting XGBoost to the Training set 
#accuracy=0.8107899731992727 std=0.011912047519692425
import xgboost as xgb
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)



# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation for Xgboost
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

