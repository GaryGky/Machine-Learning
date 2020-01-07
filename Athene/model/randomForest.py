import pandas as pd

train = pd.read_csv('final_train.csv')
train_Y = train.pop('标签')
train.drop('用户标识', axis=1, inplace=True)

test = pd.read_csv('final_test.csv')
res = test['用户标识']
test.drop('标签', inplace=True, axis=1)
test.drop('用户标识', inplace=True, axis=1)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, random_state=33, criterion='gini',verbose=True)
clf.fit(train, train_Y)

res['predict_score'] = clf.predict_proba(test)[:, 1]
print(res.head())
res.to_csv('RF_res.csv')
print(clf.feature_importances_)  # 查看特征的重要性
