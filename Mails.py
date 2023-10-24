import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
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

print('started')

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1000, activation='relu', input_shape=(3000,)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2500, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2500, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.summary()
# model.fit(X_train, y_train, epochs = 20)
# accuracy = model.evaluate(X_test, y_test)[1]
# print('Accuracy:', accuracy)

def plot_history(history):
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#   ax1.plot(history.history['loss'], label='loss')
#   ax1.plot(history.history['val_loss'], label='val_loss')
#   ax1.set_xlabel('Epoch')
#   ax1.set_ylabel('Binary crossentropy')
#   ax1.grid(True)

#   ax2.plot(history.history['accuracy'], label='accuracy')
#   ax2.plot(history.history['val_accuracy'], label='val_accuracy')
#   ax2.set_xlabel('Epoch')
#   ax2.set_ylabel('Accuracy')
#   ax2.grid(True)

#   plt.show()
  print('Hello World')
  
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(3000,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [1000, 2000, 3000]:
  for dropout_prob in[0, 0.2]:
    for lr in [0.01, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss = model.evaluate(X_test, y_test)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model