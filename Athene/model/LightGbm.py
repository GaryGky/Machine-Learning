import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

trainpath = "../data/train"
testpath = "../data/A"
resPath = trainpath + '/result'


def LGB_test(train_x, train_y, test_x, test_y, cate_col=None):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], early_stopping_rounds=100)
    feature_importances = sorted(zip(train_x.columns, clf.feature_importances_), key=lambda x: x[1])
    # clf.best_score_['valid_1']['binary_logloss']
    return feature_importances


def off_test_split(data):
    label = data.pop('标签')
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.15, random_state=2018)
    score = LGB_test(train_x, train_y, test_x, test_y)
    return score


def LGB_predict(train, predict):
    res = predict[['用户标识']]
    train_y = train.pop('标签')
    train_x = train.drop(['用户标识'], axis=1)
    test_x = predict.drop(['用户标识'], axis=1)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)])
    res['predicted_score'] = clf.predict_proba(test_x)[:, 1]
    res.to_csv('lgb_result.csv', index=False)


data = pd.read_csv('trainSet.csv')
test = pd.read_csv('testSet.csv')
label = pd.read_csv(trainpath + 'train_label.csv')
data = pd.concat([label, data], axis=1)
print(data.head())
test1 = pd.read_csv('A/test_profile_A.csv')
test1 = test1[['用户标识']]
test = pd.concat([test1, test], axis=1)
print(test.head())

LGB_predict(data, test)
