from parse import get_feature_values
from parse import get_crime_rates
from parse import get_feature_labels
from parse import add_ones
from parse import split_kfold

from functions import weight
from functions import predict_y
from functions import rmse
from functions import ridge_weight
from functions import linear_grad_descent
from functions import ridge_grad_descent

import numpy as np

# parse data files
features = get_feature_labels('train.txt')

train_x = get_feature_values('train.txt')
train_y = get_crime_rates('train.txt')

test_x = get_feature_values('test.txt')
test_y = get_crime_rates('test.txt')

# add ones to end of each instance
train_x = add_ones(train_x)
test_x = add_ones(test_x)

# find weight value
w = weight(train_x, train_y)

y_train_prediction = predict_y(train_x, w)
y_test_prediction = predict_y(test_x, w)

train_error = rmse(y_train_prediction ,train_y)
test_error = rmse(y_test_prediction, test_y)

print 'Calculating (Closed Form) Linear Regression Error...'
print 'Training RMSE: ', train_error
print 'Testing RMSE: ', test_error

# Calulate Closed Ridge Regression Now

# declare lambda values
lambda_values = [400, 200, 100, 50, 25, 12.5, 6.25, 3.125, 1.5626, 0.78125]

k_splits = split_kfold(train_x)
k_splits_y = split_kfold(train_y)

average_errors = []

for lambda_value in lambda_values:
    errors = []
    for i, split in enumerate(k_splits):
        test_x = k_splits[i]
        test_y = k_splits_y[i]

        # make a copy of k_splits
        temp_splits = k_splits[:]
        temp_splits.pop(i)

        temp_splits_y = k_splits_y[:]
        temp_splits_y.pop(i)

        train_x = np.array(temp_splits[0])
        train_y = np.array(temp_splits_y[0])

        # make training array with every other np.array
        for j in range(1, len(temp_splits)):
            train_x = np.vstack((train_x, temp_splits[j]))

        for j in range(1, len(temp_splits_y)):
            train_y = np.vstack((train_y, temp_splits_y[j]))

        # now do ridge regression stuff
        ridge_w = ridge_weight(train_x, train_y, lambda_value)
        prediction = predict_y(test_x, ridge_w)

        error = rmse(prediction, test_y)
        errors.append(error)

    avg_error = np.mean(errors)
    average_errors.append(avg_error)

# find the optimal lambda that generated the smallest RMSE (error) value
min_index = average_errors.index(min(average_errors))
min_lambda = lambda_values[min_index]

print ''
print 'Calulating (Closed Form) Ridge Regression...'
print 'Min-error lambda =', min_lambda

# refresh training values
train_x = get_feature_values('train.txt')
train_x = add_ones(train_x)

train_y = get_crime_rates('train.txt')

test_x = get_feature_values('test.txt')
test_x = add_ones(test_x)

test_y = get_crime_rates('test.txt')

# calculate ridge regression weight
w = ridge_weight(train_x, train_y, min_lambda)


# predict
y_train_prediction = predict_y(train_x, w)
y_test_prediction = predict_y(test_x, w)

# calculate error
train_error = rmse(y_train_prediction, train_y)
test_error = rmse(y_test_prediction, test_y)

print 'Training RMSE: ', train_error
print 'Testing RMSE: ', test_error
print ''


# Perform Linear Regression with Gradient Descent
print 'Calculating (Gradient Descent) Linear Regression... '
w = linear_grad_descent(train_x, train_y)

y_train_prediction = predict_y(train_x, w)
y_test_prediction = predict_y(test_x, w)

train_error = rmse(y_train_prediction, train_y)
test_error = rmse(y_test_prediction, test_y)

print 'Training RMSE: ', train_error
print 'Testing RMSE: ', test_error
print ''

# Perform Ridge Regression with Gradien Descent
print 'Calculating (Gradient Descent) Ridge Regression.... '

# declare lambda values
lambda_values = [400, 200, 100, 50, 25, 12.5, 6.25, 3.125, 1.5626, 0.78125]

k_splits = split_kfold(train_x)
k_splits_y = split_kfold(train_y)

average_errors = []

for lambda_value in lambda_values:
    errors = []
    print 'Calculating R. Grad Descent with lambda =', lambda_value
    for i, split in enumerate(k_splits):
        test_x = k_splits[i]
        test_y = k_splits_y[i]

        # make a copy of k_splits
        temp_splits = k_splits[:]
        temp_splits.pop(i)

        temp_splits_y = k_splits_y[:]
        temp_splits_y.pop(i)

        train_x = np.array(temp_splits[0])
        train_y = np.array(temp_splits_y[0])

        # make training array with every other np.array
        for j in range(1, len(temp_splits)):
            train_x = np.vstack((train_x, temp_splits[j]))

        for j in range(1, len(temp_splits_y)):
            train_y = np.vstack((train_y, temp_splits_y[j]))

        # now do ridge regression stuff
        ridge_w = ridge_grad_descent(train_x, train_y, lambda_value)
        prediction = predict_y(test_x, ridge_w)

        error = rmse(prediction, test_y)
        errors.append(error)

    avg_error = np.mean(errors)
    print 'Average Error is =', avg_error
    average_errors.append(avg_error)

# find the optimal lambda that generated the smallest RMSE (error) value
min_index = average_errors.index(min(average_errors))
min_lambda = lambda_values[min_index]

print ''
print 'Min-error lambda =', min_lambda

# refresh training values
train_x = get_feature_values('train.txt')
train_x = add_ones(train_x)

train_y = get_crime_rates('train.txt')

test_x = get_feature_values('test.txt')
test_x = add_ones(test_x)

test_y = get_crime_rates('test.txt')

# get ridge gradient descnet weight
w = ridge_grad_descent(train_x, train_y, min_lambda)
#w = ridge_weight(train_x, train_y, min_lambda)

# predict y values
y_train_prediction = predict_y(train_x, w)
y_test_prediction = predict_y(test_x, w)

# calculate error
train_error = rmse(y_train_prediction, train_y)
test_error = rmse(y_test_prediction, test_y)

print 'Training RMSE: ', train_error
print 'Testing RMSE: ', test_error
# print ''
