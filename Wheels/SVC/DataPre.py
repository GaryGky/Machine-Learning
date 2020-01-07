import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.model_selection import train_test_split

data = pd.read_csv("./svm_training_set.csv", index_col=False)

y_labels = DataFrame(data[['index', 'label']])
X_data = data.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.20, random_state=33)  # 这是随便划分的呀

print(y_train)
print(X_train)

train = pd.merge(X_train, y_train, on='index')
test = pd.merge(X_test, y_test, on='index')

train.to_csv("train.csv", index=None)
test.to_csv("test.csv", index=None)
