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

for column in df:
    plt.hist(df[df["class"] == 1][column], color='blue', label='Spam', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][column], color='red', label='Not Spam', alpha=0.7, density=True)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
