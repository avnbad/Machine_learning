# importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing files  Train
dataset= pd.read_csv('train_loan.csv')
x_train=dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values
y_train=dataset.iloc[:,-1].values

#Reshaping y
y_train=y_train.reshape(-1, 1)

# importing files Test
dataset1=pd.read_csv('test_loan.csv')

# importing files  Train
x_test=dataset1.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values

"""x=dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]] # temp file 
print(x_train[:,[5,6,7,8,9]])"""


# Generating dummy variables for train data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_train=LabelEncoder()
labelencoder_y_train=LabelEncoder()
x_train[:,0] = labelencoder_x_train.fit_transform(x_train[:,0].astype(str)) # to avoid error <
x_train[:,4] = labelencoder_x_train.fit_transform(x_train[:,4].astype(str))
x_train[:,1] = labelencoder_x_train.fit_transform(x_train[:,1].astype(str))
x_train[:,0]=labelencoder_x_train.fit_transform(x_train[:,0])
x_train[:,1]=labelencoder_x_train.fit_transform(x_train[:,1])
x_train[:,3]=labelencoder_x_train.fit_transform(x_train[:,3])
x_train[:,4]=labelencoder_x_train.fit_transform(x_train[:,4])
x_train[:,10]=labelencoder_x_train.fit_transform(x_train[:,10])

#Generating dummy for dependent varable 
y_train=labelencoder_y_train.fit_transform(y_train)

# Generating dummy variables for test data
labelencoder_x_test=LabelEncoder()
x_test[:,0] = labelencoder_x_test.fit_transform(x_test[:,0].astype(str))
x_test[:,4] = labelencoder_x_test.fit_transform(x_test[:,4].astype(str))
x_test[:,1] = labelencoder_x_test.fit_transform(x_test[:,1].astype(str))
x_test[:,0]=labelencoder_x_test.fit_transform(x_test[:,0])
x_test[:,1]=labelencoder_x_test.fit_transform(x_test[:,1])
x_test[:,3]=labelencoder_x_test.fit_transform(x_test[:,3])
x_test[:,4]=labelencoder_x_test.fit_transform(x_test[:,4])
x_test[:,10]=labelencoder_x_test.fit_transform(x_test[:,10])


# Taking care of missing data  of Train set  
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(x_train)
x_train=imputer.transform (x_train)

# Taking care of missing data  of Test set  
imputer=imputer.fit(x_test)
x_test=imputer.transform (x_test)


onehotencoder=OneHotEncoder(categorical_features=[0]) # need to took care of dummy variable trap
x_train=onehotencoder.fit_transform(x_train).toarray()
y_train=onehotencoder.fit_transform(y_train).toarray()

#Reshaping y_train
y_train=y_train.reshape(-1, 1)

#Test onehotencoder
x_test=onehotencoder.fit_transform(x_test).toarray()





"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x_train = sc_X.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

x_test = sc_X.fit_transform(x_test)"""

"""# fitting Decision Tree classification on Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)"""



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# predicting result
y_pred=classifier.predict(x_test)





