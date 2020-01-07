import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

trainpath = "../data/train"
testpath = "../data/A"
resPath = trainpath + '/result'

print('load bankStatement...')
user_bank_train = pd.read_csv(trainpath + '/train_bankStatement.csv')  # 银行信息
# features = ['用户标识', '流水时间', '交易类型', '交易金额', '工资收入标记']
user_bank_train['时间'] = user_bank_train['流水时间'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))

user_bank_train['year'] = user_bank_train['时间'].apply(lambda x: int(str(x)[0:4]))
user_bank_train['month'] = user_bank_train['时间'].apply(lambda x: int(str(x)[5:7]))
user_bank_train['day'] = user_bank_train['时间'].apply(lambda x: int(str(x)[8:10]))
user_bank_train['日期'] = user_bank_train['时间'].apply(lambda x: str(x)[0:10])
t = user_bank_train[(user_bank_train['流水时间'] == 0)].groupby('用户标识',as_index=False)['交易金额'].count()
print(t)
# print(user_bank_train['year'].head())
# print(user_bank_train['month'].head())
# print(user_bank_train['日期'].head())
PeopleInYear = user_bank_train.groupby(['year', '用户标识'], as_index=False)['用户标识'].agg({'一年内的用户活动': 'count'})
# print(PeopleInYear)
# PeopleInYear.to_csv('./一年内的用户数量.csv')
# print(user_bank_train.columns)
# features = ['用户标识', '流水时间', '交易类型', '交易金额', '工资收入标记']'''
user_label_train = pd.read_csv(trainpath + '/train_label.csv')

print('merging ...')
user_train = pd.merge(user_label_train, user_bank_train, how='inner', on='用户标识')
print('cal ...')

user_Transfer = user_train.groupby(['用户标识'], as_index=False)['交易金额'].agg({'每人总共交易金额': 'sum'})
# print(user_Transfer)
user_Transfer_TotalDay = user_train.groupby('用户标识', as_index=False)['日期'].agg({'每人总共交易天数': 'count'})
# print(user_Transfer_TotalDay)
user_Transfer = pd.merge(user_Transfer, user_Transfer_TotalDay, how='inner', on='用户标识')
# print(user_Transfer)
user_Transfer['每人平均每天交易金额'] = user_Transfer['每人总共交易金额'] / user_Transfer['每人总共交易天数']
# print(user_Transfer)

# user_Transfer.to_csv(resPath + '/每人每天交易金额分布.csv')
# 某人某天的交易金额情况分析
user_transfer_byday = user_train.groupby(['日期', '用户标识'], as_index=False)['交易金额'].agg(
    {'总和': 'sum', '平均': 'mean', '最大交易额': 'max', '最小交易额': 'min', '交易额方差': 'std', '交易次数': 'count'})
# 某人交易类型的分析
user_outcome = user_train[(user_train['交易类型'] == 1)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'收入金额': 'sum', '收入笔数': 'count'})
user_income = user_train[(user_train['交易类型'] == 0)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'支出金额': 'sum', '支出笔数': 'count'})

user_bank_stat = user_label_train
user_bank_stat = pd.merge(user_bank_stat, user_income, how='inner', on='用户标识')
user_bank_stat = pd.merge(user_bank_stat, user_outcome, how='inner', on='用户标识')
# print(user_bank_stat.head())
user_bank_stat.fillna(0, inplace=True)
# print(user_bank_stat)
user_bank_stat['收入支出金额差'] = user_bank_stat['收入金额'] - user_bank_stat['支出金额']
user_bank_stat['收入支出笔数'] = user_bank_stat['收入笔数'] - user_bank_stat['支出笔数']
# print(user_bank_stat.head())
# print(user_bank_stat['收入支出差'])
# 某人的工资情况分析
user_salary = user_train[(user_train['工资收入标记'] == 0)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'用户工资收入': 'sum', '用户工资收入次数': 'count'})
user_non_salary = user_train[(user_train['工资收入标记'] == 1)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'用户非工资收入': 'sum', '用户非工资收入次数': 'count'})
# print(user_salary.columns)
user_bank_stat = pd.merge(user_bank_stat, user_salary, how='inner', on='用户标识')
user_bank_stat = pd.merge(user_bank_stat, user_non_salary, how='inner', on='用户标识')
user_bank_stat['用户工资与非工资金额差'] = user_bank_stat['用户工资收入'] - user_bank_stat['用户非工资收入']
user_bank_stat['用户工资与非工资次数差'] = user_bank_stat['用户工资收入次数'] - user_bank_stat['用户非工资收入次数']
print(user_bank_stat)
# print(user_bank_stat)
user_bank_stat.to_csv(resPath + '/每人收入支出分布.csv')
'----------------------------------------------------------------------------------------------------------------------------------------'

user_inoutStat = user_bank_stat[(user_bank_stat['收入支出金额差'] > 0)].groupby('标签', as_index=False)['用户标识'].agg(
    {'收入大于支出': 'count'})
# print(user_inoutStat)
user_outinStat = user_bank_stat[(user_bank_stat['收入支出金额差'] <= 0)].groupby('标签', as_index=False)['用户标识'].agg(
    {'支出大于收入': 'count'})
# print(user_outinStat)
user_IO_Stat = pd.merge(user_outinStat, user_inoutStat, how='outer', on='标签')
user_IO_Stat.fillna(0, inplace=True)
# print(user_IO_Stat)
# user_IO_Stat.to_csv(resPath + '/收入支出与标签的分布.csv')

user_outcome_month = user_train[(user_train['交易类型'] == 1)].groupby(['用户标识', 'month'], as_index=False)['交易金额'].agg(
    {'收入金额': 'sum'})
user_income_month = user_train[(user_train['交易类型'] == 0)].groupby(['用户标识', 'month'], as_index=False)['交易金额'].agg(
    {'支出金额': 'sum'})

print('按月统计 ... ')
TransferInMonth = user_bank_train[(user_bank_train['month'] == 1)].groupby(['用户标识'], as_index=False)['交易金额'].agg(
    {'1月交易金额': 'sum'})
user_O_month = user_train[(user_train['交易类型'] == 1) & (user_train['month'] == 1)].groupby(['用户标识'], as_index=False)[
    '交易金额'].agg({'支出金额-1月': 'sum'})
user_I_month = user_train[(user_train['交易类型'] == 0) & (user_train['month'] == 1)].groupby(['用户标识'], as_index=False)[
    '交易金额'].agg({'收入金额-1月': 'sum'})
user_salary_month = \
    user_train[(user_train['工资收入标记'] == 1) & (user_train['month'] == 1)].groupby('用户标识', as_index=False)[
        '交易金额'].agg({'用户工资收入-1月': 'sum'})
user_non_salary_month = \
    user_train[(user_train['工资收入标记'] == 0) & (user_train['month'] == 1)].groupby('用户标识', as_index=False)['交易金额'].agg(
        {'用户非工资收入-1月': 'sum'})
for i in range(2, 13):
    print(i)
    tmp = user_train[(user_train['交易类型'] == 1) & (user_train['month'] == i)].groupby(['用户标识'], as_index=False)[
        '交易金额'].agg({str(i) + '月' + '支出金额': 'sum'})
    user_O_month = pd.merge(tmp, user_O_month, on='用户标识', how='inner')  # 支出

    tmp = user_train[(user_train['交易类型'] == 0) & (user_train['month'] == i)].groupby(['用户标识'], as_index=False)[
        '交易金额'].agg({str(i) + '月' + '收入金额': 'sum'})
    user_I_month = pd.merge(tmp, user_I_month, on='用户标识', how='inner')  # 收入

    tmp = user_bank_train[(user_bank_train['month'] == i)].groupby(['用户标识'], as_index=False)['交易金额'].agg(
        {str(i) + '月交易金额': 'sum'})
    TransferInMonth = pd.merge(TransferInMonth, tmp, on='用户标识', how='inner')  # 交易

    tmp = user_train[(user_train['工资收入标记'] == 1) & (user_train['month'] == i)].groupby('用户标识', as_index=False)[
        '交易金额'].agg({'用户工资收入-' + str(i) + '月': 'sum'})
    user_salary_month = pd.merge(user_salary_month, tmp, how='inner', on='用户标识')

    tmp = user_train[(user_train['工资收入标记'] == 0) & (user_train['month'] == i)].groupby('用户标识', as_index=False)[
        '交易金额'].agg({'用户非工资收入-' + str(i) + '月': 'sum'})
    user_non_salary_month = pd.merge(user_non_salary_month, tmp, how='inner', on='用户标识')

'''TransferInMonth.to_csv(resPath + '/每人月交易.csv')
user_O_month.to_csv(resPath + '/每人月支出.csv')
user_I_month.to_csv(resPath + '/每人月收入.csv')
user_salary_month.to_csv(resPath + '/每人月工资.csv')
user_non_salary_month.to_csv(resPath + '/每人月非工资.csv')'''
