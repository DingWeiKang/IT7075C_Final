import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('phishing.csv')

print(df)

print(df.columns)

print(df.isnull().sum())

df.dropna(axis=0)

y = df["class"]
X = df.drop(["class"],axis=1)

print(X)
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# print classification report
def print_report(y_test, prediction):
    print(classification_report(y_test, prediction, digits=5))

# Decision Tree
print("Decision Tree")
DecisionTree_model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
StartTime_Training = time.time()
DecisionTree_model.fit(X_train, y_train.values.ravel())
EndTime_Training = time.time()
print("Training time: ", EndTime_Training - StartTime_Training)
StartTime_Testing = time.time()
y_test_DecisionTree_pred = DecisionTree_model.predict(X_test)
EndTime_Testing = time.time()
print("Testing time: ", EndTime_Testing - StartTime_Testing)
print_report(y_test, y_test_DecisionTree_pred)

# Random Forest
print("Random Forest")
RandomForest_model = RandomForestClassifier()
StartTime_Training = time.time()
RandomForest_model.fit(X_train, y_train.values.ravel())
EndTime_Training = time.time()
print("Training time: ", EndTime_Training - StartTime_Training)
StartTime_Testing = time.time()
y_test_RandomForest_pred = RandomForest_model.predict(X_test)
EndTime_Testing = time.time()
print("Testing time: ", EndTime_Testing - StartTime_Testing)
print_report(y_test, y_test_RandomForest_pred)