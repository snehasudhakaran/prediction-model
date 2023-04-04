# Heart Disease Prediction Model

# importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## data collection and processing

## loading the csv data to a pandas dataframe

heartData = pd.read_csv("heart.csv")
# print(heartData.head())
# print(heartData.tail())
# print(heartData.shape)
# print(heartData.info())
# print(heartData.isnull().sum())
# print(heartData.describe())

# print(heartData["target"].value_counts())

## 0 --> Healthy heart
## 1 --> Defective heart

## splitting the features and target 

x= heartData.drop(columns="target" , axis=1)
y=heartData["target"]

# print(x)
# print(y)

## splitting the data into training data and testing data

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2 , random_state=2)
# print(x.shape , x_train.shape , x_test.shape)

## model training

Model = LogisticRegression()
Model.fit(x_train, y_train)

## model evaluation

x_trainPrediction = Model.predict(x_train)
trainingDataAccuracy = accuracy_score(x_trainPrediction , y_train)
print("Accuracy of our Model for heart disease prediction using train data",trainingDataAccuracy)

x_testPrediction = Model.predict(x_test)
testDataAccuracy = accuracy_score(x_testPrediction , y_test)
print("Accuracy of our Model for heart disease prediction using test data",testDataAccuracy)

## prediction system

input =(34,0,1,118,210,0,1,192,0,0.7,2,0,2)

# print(type(input))

#change the input data to a numpy array
inputDataAsNumpyArray = np.asarray(input)
# print(inputDataAsNumpyArray)
inputDataReshaped = inputDataAsNumpyArray.reshape(1,-1)
# print(inputDataReshaped)
Prediction = Model.predict(inputDataReshaped)
# print(Prediction)

if Prediction==0:
    print("Healthy heart (normal)")
else:
    print("defective heart")


