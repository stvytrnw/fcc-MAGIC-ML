import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)

df['class'] = (df['class'] == 'g').astype(int)

# for label in cols[:-1]:
#     plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
#     plt.title(label)
#     plt.xlabel(label)
#     plt.ylabel('Probability')
#     plt.legend()
#     plt.show()
    
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
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