import numpy as np
import sklearn as sk
from sklearn import datasets # why do I need this?
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def k_means(data, ncluster, n_iteration):
    # choose centroids randomly
    centroid_indices = np.random.choice(len(data), ncluster, replace=False)
    centroids = data[centroid_indices]
    print(centroids)
    # start assignment
    for i in range(n_iteration):
        # calculates the euclidean distances for every data point
        distances = cdist(data, centroids, 'euclidean')
        # assigns every data point to its nearest centroid
        points = np.array([np.argmin(i) for i in distances])
        # updates centroids
        for i in range(ncluster):
            centroid_temp = data[points==i].mean(axis=0)
            centroids[i] = centroid_temp
    # return the results
    return points

if __name__ == "__main__":
    # iris = sk.datasets.load_iris()
    # X = iris.data[:, 0:2] # taking only the first 2 dimensions
    # y = iris.target
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.show()

    # create ndarray with test data
    data = np.array([[1, 2], [1, 1], [2, 2], [3, 7], [3, 8], [4, 8], [7, 1],
    [7, 2], [8, 2]])
    n_cluster = 3
    n_iteration = 1000
    points_to_clusters = k_means(data, n_cluster, n_iteration)
    print(points_to_clusters)
