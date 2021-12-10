# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Srinivas Kini - skini
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    def __init__(self, n_neighbors=5, weights='uniform', metric='l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        self._X = X
        self._y = y
        # self._X.to_csv()

    # https://www.geeksforgeeks.org/find-the-most-frequent-value-in-a-numpy-array/
    # https://www.geeksforgeeks.org/how-to-get-the-indices-of-the-sorted-array-using-numpy-in-python/
    # Pseudocode from : https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
    def predict(self, X):

        def compute_weights():
            nonlocal k_neighbors
            k_neighbors = np.where(k_neighbors == 0, 1e-20, k_neighbors)  # replace 0s with a small value
            weights = 1 / k_neighbors
            return weights / sum(weights)

        predictions = np.zeros(X.shape[0])
        k = self.n_neighbors

        for idx, test_row in enumerate(X):
            distance_vector = np.array([self._distance(train_row, test_row) for train_row in self._X])
            k_neighbors = np.argsort(distance_vector)[:k]
            labels = np.array([self._y[n] for n in k_neighbors])
            weights_vector = compute_weights() if self.weights == 'distance' else None
            predictions[idx] = np.bincount(labels.astype(int),
                                           weights=weights_vector).argmax()  # Takes care of the 'votes'

        return predictions
