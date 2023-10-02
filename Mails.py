import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("emails.csv")

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

print(len(train[train['Prediction'] == 0]))
print(len(train[train['Prediction'] == 1]))

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[1:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X , y = ros.fit_resample(X,y)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
train, X_valid, y_valid = scale_dataset(valid, oversample=False)
train, X_test, y_test = scale_dataset(test, oversample=False)

# kNN 

# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train, y_train)
# y_pred = knn_model.predict(X_test)

# print(classification_report(y_test, y_pred))

# Naive Bayes

# nb_model = GaussianNB()
# nb_model = nb_model.fit(X_train, y_train)

# y_pred = nb_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# Log Regression 

log_model = LogisticRegression()
log_model = log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
print(classification_report(y_test, y_pred))
