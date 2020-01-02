# Author By Gary1111

import pandas as pd
import xgboost as xgb
from pandas import DataFrame

train = pd.read_csv("train_XY.csv", index_col=False)
test = pd.read_csv('test.csv', index_col=False)
test.rename(columns={"Unnamed: 0": 'id'}, inplace=True)
test['id'] = list(range(1, 10001))
train_x = train.drop(['id', 'label'], axis=1)  # 得到训练特征
train_y = train['label']
print(train_x.shape, train_y.shape)
print(train_x.columns)
print(train_y)

res = DataFrame()
res['id'] = test['id']
test.drop('id', axis=1, inplace=True)

clf = xgb.XGBClassifier(learning_rate=0.01,
                        n_estimators=100,
                        max_depth=4,
                        min_child_weight=6,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.005,
                        objective='multiclass',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], early_stopping_rounds=100)
res['label'] = clf.predict(test)
print(res.head(5))
res.to_csv("submission.csv", index=False)
