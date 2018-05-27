import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def loadDataset(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    return df

def collectAttributes(df):
    columns = list(df.columns.values)
    return columns

def pca(df, columns, components):
    X = df[columns[:-1]]
    y = df[columns[-1]]
    y.replace({b'no': 2.0, b'yes': 4.0, b'false': 2.0, b'true': 4.0}, inplace=True)
    X = X.fillna(X.mean())
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
    w = [eig_pairs[i][1] for i in range(components)]
    matrix_w = np.array(w).T
    new_points_projected = X_std.dot(matrix_w)
    new_space = pd.DataFrame(new_points_projected)
    new_space['defect'] = y
    return new_space

def useKnnToGetAccuracy(new_space, dimensions):
    X = np.array(new_space.ix[:, 0:dimensions])  # end index is exclusive
    y = np.array(new_space['defect'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return 100 * accuracy_score(y_test, pred)


def main():
    filename = ['kc2.arff','jm1.arff']
    components = [1, 5, 10, 15, 20]
    for files in filename:
        for dimensions in components:
            df = loadDataset(files)
            attributes = collectAttributes(df)
            new_space = pca(df, attributes, dimensions)
            accuracy = useKnnToGetAccuracy(new_space, dimensions)
            print ("For " + files + " with " + str(dimensions) + " dimensions, the accuracy was: " + str(accuracy) + "%")


if __name__ == "__main__":
    main()



