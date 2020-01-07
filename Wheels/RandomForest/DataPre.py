import pandas as pd
from sklearn.cross_validation import train_test_split

data = pd.read_csv('data.csv', index_col=False)
print(data.shape)
train_y = data[['index', 'label']]

X_train, X_test, y_train, y_test = train_test_split(data, train_y, test_size=0.3, random_state=0)
train = pd.merge(X_train, y_train, on=['index', 'label'])
test = pd.merge(X_test, y_test, on=['index', 'label'])
print(train)
print(test)

train.to_csv('train.csv', index=None)
test.to_csv('test.csv', index=None)