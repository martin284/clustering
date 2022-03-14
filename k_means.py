import numpy as np
import sklearn as sk
from sklearn import datasets # why do I need this?
import random
import matplotlib.pyplot as plt

def assign_data_points_to_clusters(data, centroids, ncluster):
    # data structure for saving data points in clusters
    clusters = [[] for i in range(ncluster)]
    # assign every data point to a cluster
    for data_point in data:
        nearest_centroid_index = 0
        min_dist = np.linalg.norm(centroids[nearest_centroid_index]-data_point)
        # for each centroid
        for i in range(ncluster):
            dist = np.linalg.norm(centroids[i]-data_point)
            if dist < min_dist:
                nearest_centroid_index = i
                min_dist = dist
        clusters[nearest_centroid_index].append(data_point)
    return clusters

def compute_centroids(clusters):
    new_centroids = []
    for i in range(len(clusters)):
        data_points = clusters[i]
        temp = np.mean(data_points, axis=0)
        new_centroids.append(temp)
    return new_centroids

def k_means(data, ncluster, n_iteration):
    # choose centroids randomly
    centroid_indices = np.random.choice(len(data), ncluster, replace=False)
    centroids = []
    for i in centroid_indices:
        centroids.append(data[i])
    # start assignment
    for i in range(n_iteration):
        clusters = assign_data_points_to_clusters(data, centroids, ncluster)
        centroids = compute_centroids(clusters)
    # return the results
    return clusters

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
    clusters = k_means(data, n_cluster, n_iteration)
    print(clusters)
