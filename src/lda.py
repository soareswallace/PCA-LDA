import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def loadDataset(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    return df

def collectAttributes(df):
    columns = list(df.columns.values)
    return columns

def lda(df, columns):
    X = df[columns[:-1]]
    y = df[columns[-1]]
    y.replace({b'no': 2.0, b'yes': 4.0, b'false': 2.0, b'true': 4.0}, inplace=True)
    X = X.fillna(X.mean())
    X_std = StandardScaler().fit_transform(X)
    np.set_printoptions(precision=4)
    mean_vectors = []

def main():
    filename = ['kc2.arff','jm1.arff']
    components = [1, 5, 10, 15, 20]
    results = {'kc2.arff': [], 'jm1.arff': []}
    for files in filename:
        for dimensions in components:
            df = loadDataset(files)
            attributes = collectAttributes(df)
            lda(df, attributes)

if __name__ == "__main__":
    main()
