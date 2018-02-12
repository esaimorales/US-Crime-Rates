from parse import get_feature_values
from parse import get_crime_rates
from parse import get_feature_labels

from functions import weight

# parse data files
features = get_feature_labels('train.txt')

train_x = get_feature_values('train.txt')
train_y = get_crime_rates('train.txt')

test_x = get_feature_values('test.txt')
text_y = get_crime_rates('test.txt')

# verify proper parsing
# print train_x
# print train_x.shape[0]
# print train_x.shape[1]
#
# print train_y
# print train_y.shape[0]
# print train_y.shape[1]

# find weight value
w = weight(train_x, train_y)
