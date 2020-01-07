import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot(data_all):
    n = data_all['总行为'].nunique()
    data_all = pd.get_dummies(data_all, columns=['总行为'])
    result = data_all.groupby('用户标识', as_index=False)['ls'].agg({'ls': 'max'})
    for i in range(n):
        print(i)
        str_temp = '总行为_' + str(i)
        tmp = data_all.groupby('用户标识', as_index=False)[str_temp].agg({str_temp: 'sum'})
        result = pd.merge(result, tmp, on='用户标识', how='left')
    result.to_csv('data/one_hot_data.csv', index=False)
    return result.drop('ls', axis=1)


print('begin read')
train = pd.read_csv("train/train_behaviors.csv")
test = pd.read_csv('../problems/problem_1/B/test_behaviors_B.csv')
train['ls'] = 1
test['ls'] = 0
print('read over')
data = pd.concat([train, test])
behavior_column = ['行为类型', '子类型1', '子类型2']
data['总行为'] = data['行为类型'].apply(str) + '-' + data['子类型1'].apply(str) + '-' + data['子类型2'].apply(str)
print(len(data['总行为'].unique()))
data['总行为'] = LabelEncoder().fit_transform(data['总行为'])
print(len(data['总行为'].unique()))
print('fit transform over')
data = data.drop(behavior_column, axis=1)
data = data.drop('星期几', axis=1)
day_cnt = data.groupby('用户标识', as_index=False)['日期'].agg({'day_cnt': lambda x: x.nunique()})
print('day_cnt over')
behavior_all_cnt = data.groupby('用户标识', as_index=False)['总行为'].agg({'behavior_all_cnt': 'count'})
day_behavior_all_mean = data.groupby(['用户标识', '日期'], as_index=False)['总行为'].agg({'count': 'count'})
day_behavior_all_mean = day_behavior_all_mean.groupby('用户标识', as_index=False)['count']\
                         .agg({'day_behavior_all_mean': 'mean'})
print('day_behavior_all_mean over')
day_behavior_all_var = data.groupby(['用户标识', '日期'], as_index=False)['总行为'].agg({'count': 'count'})
day_behavior_all_var = day_behavior_all_var.groupby('用户标识', as_index=False)['count']\
                         .agg({'day_behavior_all_var': 'var'})
print('day_behavior_all_var over')
day_behavior_mean = data.groupby(['用户标识', '日期', '总行为'], as_index=False)['总行为'].agg({'count': 'count'})
day_behavior_mean = day_behavior_mean.groupby(['用户标识', '日期'], as_index=False)['count']\
                         .agg({'mean': 'mean'})
day_behavior_mean = day_behavior_mean.groupby('用户标识', as_index=False)['mean']\
                         .agg({'day_behavior_mean': 'mean'})

print('day_behavior_mean over')
final = data.groupby('用户标识', as_index=False)['ls'].agg({'ls': 'max'})
final = pd.merge(final, day_cnt, on='用户标识')
final = pd.merge(final, behavior_all_cnt, on='用户标识')
final = pd.merge(final, day_behavior_all_mean, on='用户标识')
final = pd.merge(final, day_behavior_all_var, on='用户标识')
final = pd.merge(final, day_behavior_mean, on='用户标识')
final.to_csv('data/behavior_final.csv', index=False)
print('final merge over')
print(final)

final = pd.read_csv('data/behavior_final.csv')
behavior = one_hot(data)
print('one_hot over')

final = pd.merge(final, behavior, on='用户标识')
print('merge over')
final.to_csv('model/behavior.csv', index=False)
final_train = final[final['ls'] == 1]
final_train = final_train.drop('ls', axis=1)
final_test = final[final['ls'] == 0]
final_test = final_test.drop('ls', axis=1)
final_train.to_csv('model/behavior_train.csv', index=False)
final_test.to_csv('model/behavior_test_B.csv', index=False)


final = pd.read_csv('data/behavior_final.csv')
behavior = pd.read_csv('data/one_hot_data.csv')
behavior = behavior.drop('ls', axis=1)
print('one_hot over')

final = pd.merge(final, behavior, on='用户标识')
print('merge over')
final.to_csv('model_data/behavior.csv', index=False)
final_train = final[final['ls'] == 1]
final_train = final_train.drop('ls', axis=1)
final_test = final[final['ls'] == 0]
final_test = final_test.drop('ls', axis=1)
final_train.to_csv('model_data/behavior_train.csv', index=False)
final_test.to_csv('model_data/behavior_test.csv', index=False)