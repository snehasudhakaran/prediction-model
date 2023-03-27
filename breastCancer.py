import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# breastcancerdataset is a variable in which data been stored
breastCancerDataset = sklearn.datasets.load_breast_cancer()
# print(breastCancerDataset)

#loading data to dataframe 
dataFrame = pd.DataFrame(breastCancerDataset.data, columns = breastCancerDataset.feature_names)

#analyzing data by selecting first five rows
print(dataFrame.head())
dataFrame["label"] = breastCancerDataset.target
print(dataFrame.tail())
print(dataFrame.info())

#checking for missing values
dataFrame.isnull().sum()
#statistical measures about the data
dataFrame.describe()


dataFrame["label"].value_counts()

dataFrame.groupby("label").mean()

#separating the features and target
x= dataFrame.drop(columns='label', axis=1)
y= dataFrame["label"]

#splitting the data into training data and testing testing data
x_train, x_test, y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=2)
print(x.shape , x_train.shape, x_test.shape)

#model training
#logistic Regression

model= LogisticRegression()

#training the Logistic Regression model using training data
model.fit(x_train , y_train)

#accuracy on training data
x_trainPrediction = model.predict(x_train)
trainingDataAccuracy = accuracy_score(y_train , x_trainPrediction)
print("Accuracy of our model by training with breast cancer dataset =" , trainingDataAccuracy)

#accuracy on test data
x_testPrediction = model.predict(x_test)
testDataAccuracy = accuracy_score(y_test, x_testPrediction)
print("accuracy on test data" , testDataAccuracy)

#predictive system
Input = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902
)

inputAsNumpyArray = np.asarray(Input)
inputReshaped = inputAsNumpyArray.reshape(1,-1)

prediction = model.predict(inputReshaped)
print(prediction)

if (prediction==0):
    print("The breast cancer is Malignant")
else:
    print("The Breast cancer is Benign")