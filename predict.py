from parse import get_feature_values
from parse import get_crime_rates
from parse import get_feature_labels
from parse import add_ones

from functions import weight
from functions import predict_y
from functions import rmse
# parse data files
features = get_feature_labels('train.txt')

train_x = get_feature_values('train.txt')
train_y = get_crime_rates('train.txt')

test_x = get_feature_values('test.txt')
test_y = get_crime_rates('test.txt')

# add ones
train_x = add_ones(train_x)
test_x = add_ones(test_x)

# find weight value
w = weight(train_x, train_y)

y_train_prediction = predict_y(train_x, w)
y_test_prediction = predict_y(test_x, w)
# print test_y

train_error = rmse(y_train_prediction ,train_y)
test_error = rmse(y_test_prediction, test_y)

print train_error
print test_error
