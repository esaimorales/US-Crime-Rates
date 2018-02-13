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

# return ridge weight
# def ridge_weigth(X, Y):
#     a =
