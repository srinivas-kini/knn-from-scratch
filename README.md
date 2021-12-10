## K-Nearest Neighbors Classification

### Implementing the functions in `utils.py`

#### Euclidean Distance `euclidean_distance`

- This returns the Euclidean Distance between two vectors. Here, we make use of `numpy.linalg.norm` to calculate this
  result. Mathematically, given two vectors `X = {x1, ... ,xn} and y = {y1, ... ,yn}`, this is calculated
  as `D(X,Y) = sqrt((x1 - y1)^ 2 +...+ (xn - yn)^ 2)`.

#### Manhattan Distance `manhattan_distance`

- This returns the Manhattan Distance between two vectors. Mathematically, given two
  vectors `X = {x1, ... ,xn} and y = {y1, ... ,yn}`, this is calculated as `D(X,Y) = (|x1-y1| +...+ |xn - yn|)`.

### Implementing K Nearest Neighbors(KNN)  `k_nearest_neighbors.py`

#### The `fit` method

- In this case, there is no explicit transformation to the data, the 'training' phase simply involves assigning the
  respective attributes to the training data parameters `self._X, self_y`.

#### The `predict` method

- Once we have the testing data, we iterate over it and calculate the distance metric for _each_ row of the training
  data. We store the distances in an array `distance_vector`.
- To calculate the K nearest neighbors, we need the indices of the distances with the minimum distance from the testing
  data. This is done using `np.argsort(distance_vector)[:k]`.
- We then compare the indices with the labels present in the labeled output `self._y` and calculate the most probable (
  most common) value with `np.bincount(labels).argmax()`.
- All the predicted labels are cached in the array `predictions` and returned.
