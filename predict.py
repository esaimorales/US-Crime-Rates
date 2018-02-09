from parse import parse_file
from parse import get_features

train = parse_file('train.txt')
test = parse_file('test.txt')

features = get_features('train.txt')

print features
