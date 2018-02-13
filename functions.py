import numpy as np

def print_shape(M):
    # print 'Matrix Height: ', M.shape[0]
    # print 'Matrix Widht: ', M.shape[1]
    print '(', M.shape[0], 'x', M.shape[1], ')'


# returns w (weight-value)
def weight(X, Y):
    a = np.dot(X.T, X)
    # take inverse
    a = np.linalg.inv(a)
    b = np.dot(X.T, Y)
    w = np.dot(a,b)

    return w

# returns y-value
def predict_y(X, w):
    w = w.T
    y = np.dot(w, (X.T))
    return y.T

# return RMSE value
def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

# returns identity covariance matrix
def get_id_matrix(matrix):
    # get height
    size = matrix.shape[1]
    mtx = np.zeros((size, size))
    for i in range(size):
        mtx[i,i] = float(1)
    return mtx

# return ridge weight
def ridge_weight(X, Y, l):
    a = np.dot(X.T, X)

    a2 = l * get_id_matrix(X)
    # take inverse
    a = np.linalg.inv(a + a2)
    b = np.dot(X.T, Y)

    w = np.dot(a, b)

    return w


def linear_grad_descent(x, y):

    # set alpha
    alpha = 0.00001
    w_size = x.shape[1]

    return 0
