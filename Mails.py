import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("emails.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# print('kNN')
# model = KNeighborsClassifier(n_neighbors=5)
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('Naive Bayes')
# model = GaussianNB()
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('Log Regression')
# model = LogisticRegression(max_iter=1000)
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('Random Forest Regressor')
# model = RandomForestRegressor()
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('SVM')
# model = svm.SVC(kernel='linear')
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('NEURAL NETWORK')
# model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=[10000, 25000, 25000])
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()