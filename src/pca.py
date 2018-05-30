import pandas as pd
import numpy as np
from scipy.io import arff
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
    for i in eig_pairs:
        print(i[0])
    w = [eig_pairs[i][1] for i in range(components)]
    matrix_w = np.array(w).T
    new_points_projected = X_std.dot(matrix_w)
    new_space = pd.DataFrame(new_points_projected)
    new_space['defect'] = y
    return new_space

def useKnnToGetAccuracy(new_space, dimensions):
    X = np.array(new_space.ix[:, 0:dimensions])
    y = np.array(new_space['defect'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return 100 * accuracy_score(y_test, pred)

def showGraph(components, results, filename):
    np.random.seed(19680801)
    fig1, ax1 = plt.subplots()
    ax1.plot(components,results[filename[0]] , '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    ax1.grid()
    ax1.set_xlabel('Componentes')
    ax1.set_ylabel('Acurácia')
    ax1.set_title(filename[0])
    fig2, ax2 = plt.subplots()
    ax2.plot(components, results[filename[1]], '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    ax2.grid()
    ax2.set_xlabel('Componentes')
    ax2.set_ylabel('Acurácia')
    ax2.set_title(filename[1])
    plt.show()


def main():
    filename = ['kc2.arff','kc1.arff']
    components = [1, 5, 10, 15, 20]
    results = {'kc2.arff': [], 'kc1.arff': []}
    for files in filename:
        for dimensions in components:
            df = loadDataset(files)
            attributes = collectAttributes(df)
            new_space = pca(df, attributes, dimensions)
            accuracy = useKnnToGetAccuracy(new_space, dimensions)
            print ("For " + files + " with " + str(dimensions) + " dimensions, the accuracy was: " + str(accuracy) + "%")
            results[files].append(accuracy)
    showGraph(components, results, filename)

if __name__ == "__main__":
    main()



