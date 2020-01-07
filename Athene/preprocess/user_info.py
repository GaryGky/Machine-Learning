import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trainpath = "E:/Athene/data/train"
testpath = "E:/Athene/data/A"
resPath = trainpath + '/result'

print('load profile...')
user_info_train = pd.read_csv(trainpath + "/train_profile.csv")
# features = ['用户标识', '性别', '职业', '教育程度', '婚姻状态', '户口类型']

print('load credit...')
user_credit_train = pd.read_csv(trainpath + "/train_creditBill.csv")
# features = ['用户标识', '银行标识', '账单时间戳', '上期账单金额', '上期还款金额', '本期账单余额', '信用卡额度', '还款状态']
user_credit_train['时间'] = user_credit_train['账单时间戳'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
# print(user_credit_train['时间'])

# user_behavior_train = pd.read_csv(trainpath + '/train_behaviors.csv') # 记录行为的时间信息
# print(user_behavior_train.columns)
# features = []
print('load bankStatement...')
user_bank_train = pd.read_csv(trainpath + '/train_bankStatement.csv')  # 银行信息
user_bank_train['时间'] = user_bank_train['流水时间'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
# print(user_bank_train.columns)
# features = ['用户标识', '流水时间', '交易类型', '交易金额', '工资收入标记']

print('load label...')
user_label_train = pd.read_csv(trainpath + '/train_label.csv')
# print(user_label_train.columns)
# features = ['用户标识', '标签']

print('merging ...')
user_train = pd.merge(user_label_train, user_info_train, how='inner', on='用户标识')  # 合并取交集，不在交集中的信息将被剔除
print(user_train.columns)
# user_Train.to_csv(resPath + '/res_info_label.csv')

'''user_info_test = pd.read_csv(testpath + '/test_profile_A.csv')
user_credit_test = pd.read_csv(testpath + 'test_creditBill_A.csv')
user_behavior_test = pd.read_csv(testpath + '/test_behaviors_A.csv')
user_bank_test = pd.read_csv(testpath + '/test_bankStatement_A.csv')
user_test = pd.merge(user_info_test, user_info_train, how='inner', on='用户标识')'''

genderDistr = user_train.groupby('性别', as_index=False)['标签'].agg({'逾期': 'sum', '总数': 'count'})
genderDistr['性别逾期比'] = genderDistr['逾期'] / genderDistr['总数']
# print(genderDistr)

eduDistr = user_train.groupby('教育程度', as_index=False)['标签'].agg({'逾期': 'sum', '总数': 'count'})
eduDistr['逾期教育程度比'] = eduDistr['逾期'] / eduDistr['总数']

marDistr = user_train.groupby('婚姻状态', as_index=False)['标签'].agg({'逾期': 'sum', '总数': 'count'})
marDistr['逾期婚姻状况比'] = marDistr['逾期'] / marDistr['总数']

rprDistr = user_train.groupby('户口类型', as_index=False)['标签'].agg({'逾期': 'sum', '总数': 'count'})
rprDistr['逾期户口类型比'] = rprDistr['逾期'] / rprDistr['总数']

jobDistr = user_train.groupby('性别', as_index=False)['标签'].agg({'逾期': 'sum', '总数': 'count'})
jobDistr['逾期性别比'] = jobDistr['逾期'] / jobDistr['总数']
### 用户的基本信息
