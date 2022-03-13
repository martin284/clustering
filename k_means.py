import numpy as np
import sklearn as sk
from sklearn import datasets # why do I need this?

def assign_data_points_to_clusters(data, centroid_indices):
    # data structure for saving data points in clusters
    clusters = {}
    for index in centroid_indices:
        clusters[index] = []
    # assign every data point to a cluster
    for data_point in data:
        nearest_centroid_index = centroid_indices[0]
        min_distance = np.linalg.norm(data[nearest_centroid_index]-data_point)
        for centroid_index in centroid_indices:
            distance = np.linalg.norm(data[centroid_index]-data_point)
            if distance < min_distance:
                nearest_centroid_index = centroid_index
                min_distance = distance
        clusters[nearest_centroid_index].append(data_point)
    return clusters



def k_means(data, ncluster):
    # create centroids via choosing indices randomly
    # not random yet for testing
    centroid_indices = [0, 3, 6]
    clusters = assign_data_points_to_clusters(data, centroid_indices)
    print(clusters)

if __name__ == "__main__":
    # iris = sk.datasets.load_iris().data
    # print(iris)
    # print(type(iris))

    # create ndarray with test data
    data = np.array([[1, 2], [1, 1], [2, 2], [3, 7], [3, 8], [4, 8], [7, 1],
    [7, 2], [8, 2]])

    k_means(data, 3)
