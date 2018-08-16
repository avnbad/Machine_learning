# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')


#droping coloumns
drop_list=['Alley','PoolQC','Fence','MiscFeature']
dataset=dataset.drop(drop_list, axis = 1, inplace = False)

X = dataset.iloc[:, 1:-1]
y=  dataset.iloc[:, [-1]]



# Generating dummy variables for  data
X = pd.get_dummies(X,drop_first=True)

# Taking care of missing data    
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X)
X=imputer.transform (X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# predicting result
y_pred=regressor.predict(X_test)

#Comparing the result
regressor.score(X_train, y_train)    #.966794

regressor.score(X_test, y_test)     #.815554 clearly there is some overfitting

#ploting the curves
from sklearn.model_selection import learning_curve
tsz = np.linspace(0.1, 1, 10)
train_sizes, train_scores, test_scores = learning_curve(regressor, X, y, train_sizes=tsz)

fig = plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'ro-', label="Train Scores")
plt.plot(train_sizes, test_scores.mean(axis=1), 'go-', label="Test Scores")
plt.title('Learning Curve: Random Forest Tree')
plt.ylim((0.5, 1.0))
plt.legend()
plt.draw()
plt.show()



