# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Srinivas Kini - skini
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
import numpy as np

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def manhattan_distance(x1, x2):
    return np.sum(np.array([np.abs(i - j) for i, j in zip(x1, x2)]))
