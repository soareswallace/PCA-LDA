import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def loadDataset(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    return df

def collectAttributes(df):
    columns = list(df.columns.values)
    return columns

def useKnnToGetAccuracy(new_space, dimensions):
    X = np.array(new_space.ix[:, 0:dimensions])  # end index is exclusive
    y = np.array(new_space['defect'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return 100 * accuracy_score(y_test, pred)

def lda(df, columns, components):
    y = df[columns[-1]]
    X = df.iloc[:, :-1].values
    y.replace({b'no': 2.0, b'yes': 4.0, b'false': 2.0, b'true': 4.0}, inplace=True)
    is_nan = ~np.isnan(X).any(axis=1)
    X = X[is_nan]
    y = y[is_nan]
    number_of_features = len(columns)-1
    mean_vec = []
    for i in df["defects"].unique():
        mean_vec.append(np.array((df[df["defects"] == i].mean()[:number_of_features])))
    S_W = np.zeros((number_of_features, number_of_features))
    for cl, mv in zip(range(1, number_of_features), mean_vec):
        class_sc_mat = np.zeros((number_of_features, number_of_features))  # scatter matrix for every class
        for row in X[y == cl]:
            row = row.reshape(number_of_features, 1)
            mv = mv.reshape(number_of_features, 1)  # make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat  # sum class scatter matrices

    S_B = np.zeros((number_of_features, number_of_features))
    for i, mean_vec in enumerate(mean_vec):
        n = X[y == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(number_of_features, 1)  # make column vector
        overall_mean = np.zeros((number_of_features, 1))
        overall_mean = overall_mean.reshape(number_of_features, 1)  # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    for i in range(len(eig_vals)):
        eigv = eig_vecs[:, i].reshape(number_of_features, 1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                             eig_vals[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    w = [eig_pairs[i][1] for i in range(components)]
    matrix_w = np.array(w).T
    new_points_projected = X.dot(matrix_w)
    new_space = pd.DataFrame(new_points_projected)
    new_space['defect'] = y
    return new_space


def main():
    filename = ['kc2.arff','kc1.arff']
    components = [1]
    results = {'kc2.arff': [], 'kc1.arff': []}
    for files in filename:
        for dimensions in components:
            df = loadDataset(files)
            attributes = collectAttributes(df)
            new_space = lda(df, attributes, dimensions)
            accuracy = useKnnToGetAccuracy(new_space, dimensions)
            print("The accuracy was: " + str(accuracy) + "%")
            results[files].append(accuracy)

if __name__ == "__main__":
    main()
