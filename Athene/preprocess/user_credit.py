import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trainpath = "../data/train"
testpath = "../data/A"
resPath = trainpath + '/result'

print('load credit...')
user_credit_train = pd.read_csv(trainpath + "/train_creditBill.csv")
# features = ['用户标识', '银行标识', '账单时间戳', '上期账单金额', '上期还款金额', '本期账单余额', '信用卡额度', '还款状态']
user_credit_train['时间'] = user_credit_train['账单时间戳'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
# print(user_credit_train['时间'])
user_credit_train['year'] = user_credit_train['时间'].apply(lambda x: int(str(x)[0:4]))
user_credit_train['month'] = user_credit_train['时间'].apply(lambda x: int(str(x)[5:7]))
user_credit_train['day'] = user_credit_train['时间'].apply(lambda x: int(str(x)[8:10]))
user_credit_train['日期'] = user_credit_train['时间'].apply(lambda x: str(x)[0:10])
print('load label...')

user_label_train = pd.read_csv(trainpath + '/train_label.csv')

print('merging ...')

user_credit = pd.merge(user_label_train, user_credit_train, how='inner', on='用户标识')

print('cal ...')

user_credit['上期未还款金额'] = user_credit['上期账单金额'] - user_credit['上期还款金额']
user_credit['本期账单金额'] = user_credit['信用卡额度'] - user_credit['本期账单余额']
user_credit['相邻两期的账单金额差'] = user_credit['本期账单金额'] - user_credit['上期账单金额']
user_credit['本期还款总额'] = user_credit['上期账单金额'] - user_credit['上期还款金额'] + user_credit['本期账单金额']
user_credit['已经使用信用卡额度'] = user_credit['信用卡额度'] - user_credit['本期账单余额']
# print(user_credit.columns.tolist())
# print(user_credit.columns.tolist())
# exit()
# 信用卡数据的统计分析
# ex：用户A在1970.9月的信息汇总
user_credit_sum = user_credit.loc[:,
                  ['用户标识', 'month', 'year', '上期账单金额', '上期还款金额', '本期账单余额', '上期未还款金额', '本期账单金额', '相邻两期的账单金额差', '本期还款总额',
                   '已经使用信用卡额度']].groupby(
    ['用户标识', 'year', 'month'], as_index=False).sum()
print(user_credit_sum.head(20))
print(user_credit_sum.shape)
print(user_credit.shape)
user_credit.to_csv(resPath + '/用户信用卡信息分析.csv')
