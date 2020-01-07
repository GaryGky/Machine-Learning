import pandas as pd

trainPath = '../data/train'
resPath = trainPath + '/result'


# print(user_label.columns)
user_IO = pd.read_csv(resPath + '/每人收入支出分布.csv')
# user_IO.drop('Unnamed: 0', inplace=True)
# print(user_IO.columns)

train = user_IO
print(train.columns)






