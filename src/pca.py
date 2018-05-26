import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

data = arff.loadarff('kc1.arff')
df = pd.DataFrame(data[0])
columns = list(df.columns.values)

X = df[columns[:-1]]
#print(X.head())
y = df[columns[-1]]
#print(Y.head())


X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

# print('Covariance matrix \n%s' %cov_mat)
# print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)


u,s,v = np.linalg.svd(X_std.T)
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])

#Two PCAs are chossen to project the data
matrix_w = np.hstack((eig_pairs[0][1].reshape(len(columns)-1,1),eig_pairs[1][1].reshape(len(columns)-1,1), eig_pairs[2][1].reshape(len(columns)-1,1)))
Y = X_std.dot(matrix_w)

