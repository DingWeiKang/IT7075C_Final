import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
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

y_pred1= DecisionTree_model.predict_proba(X_test)[:, 1]
auc = metrics.roc_auc_score(y_test, y_pred1)

false_positive_rate1, true_positive_rate1, thresolds1 = metrics.roc_curve(y_test, y_pred1)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Decision Tree - AUC & ROC Curve")
plt.plot(false_positive_rate1, true_positive_rate1, 'g')
plt.fill_between(false_positive_rate1, true_positive_rate1, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

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


y_pred2 = RandomForest_model.predict_proba(X_test)[:, 1]
auc = metrics.roc_auc_score(y_test, y_pred2)

false_positive_rate2, true_positive_rate2, thresolds2 = metrics.roc_curve(y_test, y_pred2)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Random Forest - AUC & ROC Curve")
plt.plot(false_positive_rate2, true_positive_rate2, 'g')
plt.fill_between(false_positive_rate2, true_positive_rate2, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()