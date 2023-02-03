import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('CySecData.csv')

print(df)

print(df.info())

df1 = pd.get_dummies(df)

print(df1)

X = df1.drop(['class_anomaly','class_normal'], axis = 1)
y = df1["class_anomaly"]

sc = StandardScaler()
X = sc.fit_transform(X)

LogisticRegression_model=LogisticRegression()
SVM_model=SVC()
RandomForest_model=RandomForestClassifier()

skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

print("Logistic Regression")
scores = cross_val_score(LogisticRegression_model, X, y.values.ravel(), scoring='accuracy', cv=skf, n_jobs=-1)
print('Accuracy Mean: ', mean(scores))
print('Accuracy std: ', std(scores))

print("SVM")
scores = cross_val_score(SVM_model, X, y.values.ravel(), scoring='accuracy', cv=skf, n_jobs=-1)
print('Accuracy Mean: ', mean(scores))
print('Accuracy std: ', std(scores))

print("Random Forest")
scores = cross_val_score(RandomForest_model, X, y.values.ravel(), scoring='accuracy', cv=skf, n_jobs=-1)
print('Accuracy Mean: ', mean(scores))
print('Accuracy std: ', std(scores))
