import numpy as np
import random

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


# Follows formula:
# W_current = W_previous + alpha*X_transpose (y - X*W_previous)
def linear_grad_descent(X, Y):

    # set alpha
    alpha = 0.00001
    w_size = X.shape[1]

    rand_val = random.uniform(0,1)

    W_previous = np.full((w_size, 1), rand_val)

    # inner = (X*W_previous)
    inner = np.dot(X, W_previous)
    parenthesis = Y - inner
    # print parenthesis, parenthesis.shape

    second = alpha * X.T
    # print second, second.shape

    outer = np.dot(second, parenthesis)
    W_current = W_previous + outer
    # print W_current

    change = 1
    converging_differential = 0.00001

    while abs(change) > converging_differential:
        W_previous = W_current
        inner = np.dot(X, W_previous)
        parenthesis = Y - inner
        second = alpha * X.T
        outer = np.dot(second, parenthesis)
        W_current = W_previous + outer

        change = np.amin(W_current - W_previous)

    return W_current
