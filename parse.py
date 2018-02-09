import numpy as np

# returns numpy array with data values, skips feature labels
def parse_file(file_name):
    with open(file_name) as f:
        next(f)
        return np.array([[float(val) for val in line.split()]for line in f])

# returns numpy array with feature labels
def get_features(file_name):
    with open(file_name) as f:
        return np.array([val for val in f.readline().split()])

# splits data k-fold
def split_kfold(array, k=5):
    i, tup, width = 0, (), array.shape[1]
    for _ in range(k):
        tup.append(array[i:width/k])
        i += width
    return tup
