import numpy as np
import random

# debugging function that prints matrix shape
def print_shape(M):
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
def rmse(prediction_values, true_values):
    return np.sqrt(((prediction_values - true_values) ** 2).mean())

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

    # create random value between 0 and 1 and set to W_t
    rand_val = random.uniform(0,1)
    W_previous = np.full((X.shape[1], 1), rand_val)

    # Calculate W ^ (t+1)
    inner = np.dot(X, W_previous)
    parenthesis = Y - inner
    second = alpha * X.T
    outer = np.dot(second, parenthesis)
    W_current = W_previous + outer

    change = 1
    converging_differential = 0.00001

    # loop until change is less than epsilon
    while abs(change) > converging_differential:
        # reset W_previous
        W_previous = W_current

        # Recalculate W_current
        inner = np.dot(X, W_previous)
        parenthesis = Y - inner
        second = alpha * X.T
        outer = np.dot(second, parenthesis)
        W_current = W_previous + outer
        change = np.amin(W_current - W_previous)

    return W_current


def ridge_grad_descent(X, Y, l):

    # set alpha
    alpha  = 0.00001

    # create random value between 0 and 1 and set to W_t
    rand_val = random.uniform(0,1)
    W_previous = np.full((X.shape[1], 1), rand_val)

    inner = np.dot(X, W_previous)
    parenthesis = Y - inner
    inner_1 = np.dot(X.T, parenthesis)
    inner_2 = l * W_previous

    big_parenthesis = inner_1 - inner_2

    W_current = W_previous + big_parenthesis

    # print W_current
    change = 1
    converging_differential = 0.00001

    while abs(change) > converging_differential:
        W_previous = W_current

        inner = np.dot(X, W_previous)
        parenthesis = Y - inner
        inner_1 = np.dot(X.T, parenthesis)
        # inner_1 = X.T * parenthesis
        inner_2 = l * W_previous

        big_parenthesis = inner_1 - inner_2
        second_term = alpha * big_parenthesis

        W_current = W_previous + second_term
        change = np.amin(W_current - W_previous)

    # print W_current
    return W_current
