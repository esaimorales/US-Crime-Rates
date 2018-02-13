import numpy as np

# returns numpy array with data values, skips feature labels
def get_feature_values(file_name):
    with open(file_name) as f:
        next(f)
        return np.array([[float(val) for val in line.split()[1:]] for line in f])

# returns numpy array with crime rate values, skips feature labels
def get_crime_rates(file_name):
    with open(file_name) as f:
        next(f)
        return np.array([[float(val) for val in line.split()[:1]] for line in f])

# returns numpy array with feature labels
def get_feature_labels(file_name):
    with open(file_name) as f:
        return np.array([val for val in f.readline().split()])

# adds a 1 to the end of every row (instance)
def add_ones(array):
    ones = np.ones((array.shape[0], 1))
    return np.hstack((array, ones))

# splits data k-fold
def split_kfold(array, k=5):
    i, tup, width = 0, (), array.shape[1]
    for _ in range(k):
        tup.append(array[i:width/k])
        i += width
    return tup
