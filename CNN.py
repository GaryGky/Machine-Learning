import pandas as pd
import xgboost as xgb
from pandas import DataFrame
from xgboost import plot_importance
import lightgbm as lgb


train = pd.read_csv("train_XY.csv", index_col=False)
test = pd.read_csv('test.csv',index_col=False)
test.rename(columns = {"Unnamed: 0" :'id'},inplace=True)
test['id'] = list(range(1,10001))
train_x = train.drop(['id','label'],axis=1) # 得到训练特征
train_y = train['label']
print(train_x.shape,train_y.shape)
print(train_x.columns)
print(train_y)

res = DataFrame()
res['id'] = test['id']
test.drop('id',axis=1,inplace=True)

