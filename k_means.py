import numpy as np
import sklearn as sk
from sklearn import datasets # why do I need this?

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

def k_means(data, ncluster):
    # create centroids via choosing indices randomly
    # not random yet for testing
    centroid_indices = [0, 3, 6]
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(data[centroid_index])

    # start assignment
    counter = 0
    while True:
        counter += 1
        clusters = assign_data_points_to_clusters(data, centroids, ncluster)
        new_centroids = compute_centroids(clusters)
        # finish loop if there are no changes anymore
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    print('The algorithm ran through ' + str(counter) + " iterations.")
    return clusters

if __name__ == "__main__":
    # iris = sk.datasets.load_iris().data
    # print(iris)
    # print(type(iris))

    # create ndarray with test data
    data = np.array([[1, 2], [1, 1], [2, 2], [3, 7], [3, 8], [4, 8], [7, 1],
    [7, 2], [8, 2]])

    clusters = k_means(data, 3)
    print(clusters)
