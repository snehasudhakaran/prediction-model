import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

diabetesData = pd.read_csv("diabetes.csv")
# print(diabetesData)
# print(diabetesData.head())
# print(diabetesData.tail())
# print(diabetesData.shape)
# print(diabetesData.info())
# print(diabetesData.describe())
# print(diabetesData["Outcome"].value_counts())
# print(diabetesData.isnull().sum())
# print(diabetesData.groupby("Outcome").mean())
x=diabetesData.drop(columns="Outcome" , axis=1)
# print(x)
y=diabetesData["Outcome"]
# print(y)
#Data Standardization
scaler = StandardScaler()
scaler.fit(x)
standardData = scaler.transform(x)
# print(standardData)

x= standardData
y= diabetesData["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, stratify=y, random_state=2)

#model training
classifier= svm.SVC(kernel="linear")
classifier.fit(x_train, y_train)

#model evaluation
x_trainingPrediction = classifier.predict(x_train)
x_trainingAccuracy = accuracy_score(x_trainingPrediction , y_train)
print(x_trainingAccuracy)
x_testingPrediction = classifier.predict(x_test)
x_testingAccuracy = accuracy_score(x_testingPrediction, y_test)
print(x_testingAccuracy)
#predictive model
input = (5,116,74,0,0,25.6,0.201,30)
inputAsNumpyArray = np.asarray(input)
inputreshape = inputAsNumpyArray.reshape(1,-1)
stdData = scaler.transform(inputreshape)
prediction = classifier.predict(stdData)
print(prediction)
if prediction==0:
    print("non diabetes")
else:
    print("diabetes")