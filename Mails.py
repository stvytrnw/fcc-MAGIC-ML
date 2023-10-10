import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

df = pd.read_csv("emails.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# def scale_dataset(X, y, oversample=False):
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
        
#     if oversample:
#         ros = RandomOverSampler()
#         X , y = ros.fit_resample(X,y)
    
#     return X, y

# X_train, y_train = scale_dataset(X_train, y_train, oversample=False)
# X_test, y_test = scale_dataset(X_test, y_test, oversample=False)

print('kNN')
model = KNeighborsClassifier(n_neighbors=5)
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('Naive Bayes')
model = GaussianNB()
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('Log Regression')
model = LogisticRegression(max_iter=1000)
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()