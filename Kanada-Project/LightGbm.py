# Author By Gary1111

import lightgbm as lgb
import pandas as pd
from pandas import DataFrame

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

clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=500, objective='multiclass',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.1, min_child_weight=25, random_state=2018, n_jobs=50
    )
clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], early_stopping_rounds=100)
res['label'] = clf.predict(test)
print(res.head(5))
res.to_csv("submission.csv",index=False)