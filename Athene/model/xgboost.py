import pandas as pd

train = pd.read_csv('final_train_1.csv')
train_y = train['标签']
train.drop(['用户标识', '标签'], axis=1, inplace=True)
test = pd.read_csv('final_test_1.csv')
res = test.pop('用户标识')
test.drop('标签', inplace=True)
print(train.shape)
print(test.shape)

from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.01,
                    n_estimators=5000,
                    max_depth=4,
                    min_child_weight=6,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.005,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

eval_set = [(train, train_y)]
xgb.fit(train, train_y, eval_metric="logloss", eval_set=eval_set, verbose=True)
res['predicted_score'] = xgb.predict_proba(test)[:, 1]
res.to_csv('xgboostRes.csv')
