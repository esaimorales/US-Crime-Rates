import numpy as np

def print_shape(M):
    print 'Matrix Height: ', M.shape[0]
    print 'Matrix Widht: ', M.shape[1]

def weight(X, Y):
    a = np.dot(X.T, X)
    # a = (X.T)*(X)
    print a
    print_shape(a)

    a = np.linalg.inv(a)
    print a
    print_shape(a)

    b = np.dot(X.T, Y)
    # b = (X.T)*(Y)
    print b
    print_shape(b)

    w = np.dot(a,b)
    # w = a*b
    print w
    print_shape(w)
    return w

def predict_y(X, w):
    w = w.T
    y = w * X
    print y
    print_shape(y)
