import numpy as np
import pandas as pd
import numpy
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

def load_dataset(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    return df

def collect_attributes(df):
    columns = list(df.columns.values)
    return columns

def pca(df, columns):

    X = df[columns[:-1]]
    y = df[columns[-1]]
    n = len(y)
    print(y.T[0])
    X_std = StandardScaler().fit_transform(X)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
    #Two PCAs are chossen to project the data

    matrix_w = np.hstack((eig_pairs[0][1].reshape(len(columns)-1,1),eig_pairs[1][1].reshape(len(columns)-1,1), eig_pairs[2][1].reshape(len(columns)-1,1)))
    new_points_projected = X_std.dot(matrix_w)
    new_space = pd.DataFrame(new_points_projected)
    new_space['defect'] = y
    print ('New Space Generated')
    return new_space


def main():
    filename = 'kc1.arff'
    df = load_dataset(filename)
    attributes = collect_attributes(df)
    new_space = pca(df, attributes)


if __name__ == "__main__":
    main()



