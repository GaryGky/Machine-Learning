import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

trainpath = "./data/train"
testpath = "./data/A"
resPath = trainpath + '/result'

print('load credit...')
user_credit_test = pd.read_csv(testpath + "/test_creditBill_B.csv")
# features = ['用户标识', '银行标识', '账单时间戳', '上期账单金额', '上期还款金额', '本期账单余额', '信用卡额度', '还款状态']
user_credit_test['时间'] = user_credit_test['账单时间戳'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
# print(user_credit_train['时间'])
user_credit_test['year'] = user_credit_test['时间'].apply(lambda x: int(str(x)[0:4]))
user_credit_test['month'] = user_credit_test['时间'].apply(lambda x: int(str(x)[5:7]))
user_credit_test['day'] = user_credit_test['时间'].apply(lambda x: int(str(x)[8:10]))
user_credit_test['日期'] = user_credit_test['时间'].apply(lambda x: str(x)[0:10])
print('load label...')
user_credit_test.groupby('year', as_index=False).count()  # 1970, 2086-2092

print('merging ...')
user_credit = user_credit_test
t = user_credit[(user_credit['账单时间戳'] == 0)].groupby('用户标识', as_index=False)['银行标识'].count()
# print(t)
bank_ID_num = user_credit.groupby('用户标识', as_index=False)['银行标识'].agg({'银行卡标识数量': 'nunique', '信用卡账单消费数量': 'count'})
# print(bank_ID_num)
print('cal ...')
# 分析基础数据
# 1.银行标识的范围
user_credit_bankLabel = user_credit.groupby(['用户标识'], as_index=False)['银行标识'].agg(
    {'用户不同银行标识的数量': 'nunique', '用户信用卡记录条数': 'count'})
user_credit_bankLabel.sort_values('用户不同银行标识的数量', inplace=True)
# print(user_credit_tmp.shape)
# print(user_credit_tmp) # 用户标识为唯一标识

# 2.分析还款状态
user_credit_repay = user_credit.groupby('还款状态', as_index=False)['用户标识'].agg({'用户还款状态分析': 'count'})
repay_credit_user = user_credit.groupby(['用户标识', '银行标识', 'month'], as_index=False)['还款状态'].agg(
    {'用户一个月内不同还款状态': 'nunique'})
# print('用户还款状态', repay_credit_user.shape)
# # 加上标签
user_repay = user_credit_test.groupby('用户标识', as_index=False)['还款状态'].agg({'还款记录总次数': 'count', '还款成功次数': 'sum'})
user_repay['还款成功率'] = user_repay['还款成功次数'] / user_repay['还款记录总次数']
user_repay.sort_values('还款成功率', inplace=True)
# print(user_repay.shape)
# print(user_repay.head(10)) # 用户唯一标识
user_credit_test = pd.merge(user_repay, user_credit_bankLabel, how='outer', on='用户标识')

user_credit_num = user_credit.groupby('用户标识', as_index=False)['银行标识'].agg({'每个用户的信用卡张数': 'nunique'})

user_credit_topMean = user_credit.groupby(['用户标识', '银行标识'], as_index=False)['信用卡额度'].agg({"平均每张信用卡额度": 'mean'})
user_credit_topMean = user_credit_topMean.groupby('用户标识', as_index=False)['平均每张信用卡额度'].agg(
    {'信用卡额度总和': 'sum', '信用卡张数': 'count', '单张信用卡的最高额度': 'max', '单张信用卡的最低额度': 'min'})
user_credit_topMean['平均每张信用卡额度'] = user_credit_topMean['信用卡额度总和'] / user_credit_topMean['信用卡张数']

user_credit_cosNum = user_credit.groupby(['用户标识', '银行标识'], as_index=False)['信用卡额度'].agg({'信用卡的消费次数': 'count'})
user_credit_cosNum = user_credit_cosNum.groupby('用户标识', as_index=False)['信用卡的消费次数'].agg(
    {'消费次数的最大值': 'max', '消费次数的最小值': 'min'})
# user_credit_cosMoney = user_credit.groupby(['用户标识', '银行标识'], as_index=False)['本期账单金额'].agg(cal_consume)
# print(user_credit_cosMoney)
user_credit_ult = pd.merge(user_credit_topMean, user_credit_cosNum, how='inner', on='用户标识')
user_credit_ult = pd.merge(user_credit_ult, user_credit_test, how='outer', on='用户标识')
# print(user_credit_ult.columns.tolist())

# 3.基本数据同期内为常数:包括还款状态
user_credit['上期未还款金额'] = user_credit['上期账单金额'] - user_credit['上期还款金额']
user_credit['本期账单金额'] = user_credit['信用卡额度'] - user_credit['本期账单余额']
user_credit['相邻两期账单金额差'] = user_credit['本期账单金额'] - user_credit['上期账单金额']
user_credit['本期还款总额'] = user_credit['上期账单金额'] - user_credit['上期还款金额'] + user_credit['本期账单金额']
user_credit['已经使用的信用卡额度'] = user_credit['信用卡额度'] - user_credit['本期账单余额']
# print(user_credit.shape)
# print(user_credit.columns.tolist())
user_credit_new1 = user_credit.groupby(['用户标识'], as_index=False)['上期未还款金额'].agg(
    {'上期未还款的最大值': 'max', '上期未还款的最小值': 'min', '上期未还款的平均值': 'mean', '上期未还款的中位数': 'median', '上期未还款的方差': 'var'})
user_credit_new2 = user_credit.groupby('用户标识', as_index=False)['本期账单金额'].agg(
    {'本期账单金额max': 'max', '本期账单金额min': 'min', '本期账单金额mean': 'mean', '本期账单金额median': 'median', '本期账单金额var': 'var'})
user_credit_new3 = user_credit.groupby('用户标识', as_index=False)['相邻两期账单金额差'].agg(
    {'相邻两期账单金额差max': 'max', '相邻两期账单金额差min': 'min', '相邻两期账单金额差mean': 'mean', '相邻两期账单金额差median': 'median',
     '相邻两期账单金额差var': 'var'})
user_credit_new4 = user_credit.groupby('用户标识', as_index=False)['本期还款总额'].agg(
    {'本期还款总额max': 'max', '本期还款总额min': 'min', '本期还款总额mean': 'mean', '本期还款总额median': 'median', '本期还款总额var': 'var'})
user_credit_new5 = user_credit.groupby('用户标识', as_index=False)['已经使用的信用卡额度'].agg(
    {'已经使用的信用卡额度max': 'max', '已经使用的信用卡额度min': 'min', '已经使用的信用卡额度mean': 'mean', '已经使用的信用卡额度median': 'median',
     '已经使用的信用卡额度var': 'var'})
user_credit_ult = pd.merge(user_credit_ult, user_credit_new1, how='inner', on='用户标识')
user_credit_ult = pd.merge(user_credit_ult, user_credit_new2, how='inner', on='用户标识')
user_credit_ult = pd.merge(user_credit_ult, user_credit_new3, how='inner', on='用户标识')
user_credit_ult = pd.merge(user_credit_ult, user_credit_new4, how='inner', on='用户标识')
user_credit_ult = pd.merge(user_credit_ult, user_credit_new5, how='inner', on='用户标识')
print(user_credit_ult.shape)
print(user_credit_ult)

'-----------------------------------------------------------------------------------------------------------------------------'
# %% 银行卡
print('load bankStatement...')
user_bank_test = pd.read_csv(testpath + '/test_bankStatement_B.csv')  # 银行信息
# features = ['用户标识', '流水时间', '交易类型', '交易金额', '工资收入标记']
user_bank_test['时间'] = user_bank_test['流水时间'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))

user_bank_test['year'] = user_bank_test['时间'].apply(lambda x: int(str(x)[0:4]))
user_bank_test['month'] = user_bank_test['时间'].apply(lambda x: int(str(x)[5:7]))
user_bank_test['day'] = user_bank_test['时间'].apply(lambda x: int(str(x)[8:10]))
user_bank_test['日期'] = user_bank_test['时间'].apply(lambda x: str(x)[0:10])

print('merging ...')
user_train = user_bank_test
print('cal ...')

user_Transfer = user_train.groupby(['用户标识'], as_index=False)['交易金额'].agg({'每人总共交易金额': 'sum'})
user_Transfer_TotalDay = user_train.groupby('用户标识', as_index=False)['日期'].agg({'每人总共交易天数': 'count'})
user_Transfer = pd.merge(user_Transfer, user_Transfer_TotalDay, how='inner', on='用户标识')
user_Transfer['每人平均每天交易金额'] = user_Transfer['每人总共交易金额'] / user_Transfer['每人总共交易天数']

# 某人某天的交易金额情况分析
user_transfer_byday = user_train.groupby(['日期', '用户标识'], as_index=False)['交易金额'].agg(
    {'总和': 'sum', '平均': 'mean', '最大交易额': 'max', '最小交易额': 'min', '交易额方差': 'std', '交易次数': 'count'})
# print(user_transfer_byday)
user_transDay_stat_1 = user_transfer_byday.groupby('用户标识', as_index=False)['总和'].agg(
    {'一天内交易金额的总和': 'sum', '一天内交易金额的方差': 'std', '一天内交易金额最大量': 'max', '一天内交易金额最小量': 'min', '一天内交易金额均值': 'mean'})
user_transDay_stat_2 = user_transfer_byday.groupby('用户标识', as_index=False)['交易次数'].agg(
    {'一天内交易次数的总和': 'sum', '一天内交易的次数方差': 'std', '一天内交易次数最大量': 'max', '一天内交易次数最小量': 'min', '一天内交易次数均值': 'mean'})
user_transDay_stat = pd.merge(user_transDay_stat_1, user_transDay_stat_2, how='outer', on='用户标识')
# print(user_transDay_stat)

# 某人交易类型的分析
user_outcome = user_train[(user_train['交易类型'] == 1)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'收入金额': 'sum', '收入笔数': 'count', '收入最大值': 'max'})
user_income = user_train[(user_train['交易类型'] == 0)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'支出金额': 'sum', '支出笔数': 'count', '支出最大值': 'max'})
user_bank_stat = pd.merge(user_outcome, user_income, how='outer', on='用户标识')
# print(user_bank_stat)
user_bank_stat.fillna(0, inplace=True)
# print(user_bank_stat)
user_bank_stat['收入支出金额差'] = user_bank_stat['收入金额'] - user_bank_stat['支出金额']
user_bank_stat['收入支出笔数'] = user_bank_stat['收入笔数'] - user_bank_stat['支出笔数']
user_bank_stat['收支最大值差'] = user_bank_stat['收入最大值'] - user_bank_stat['支出最大值']

# 某人的工资情况分析
user_salary = user_train[(user_train['工资收入标记'] == 0)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'用户工资收入': 'sum', '用户工资收入次数': 'count'})
user_non_salary = user_train[(user_train['工资收入标记'] == 1)].groupby('用户标识', as_index=False)['交易金额'].agg(
    {'用户非工资收入': 'sum', '用户非工资收入次数': 'count'})
# print(user_salary.columns)
user_bank_stat = pd.merge(user_bank_stat, user_salary, how='outer', on='用户标识')
user_bank_stat = pd.merge(user_bank_stat, user_non_salary, how='outer', on='用户标识')
user_bank_stat.fillna(0, inplace=True)
user_bank_stat['用户工资与非工资金额差'] = user_bank_stat['用户工资收入'] - user_bank_stat['用户非工资收入']
user_bank_stat['用户工资与非工资次数差'] = user_bank_stat['用户工资收入次数'] - user_bank_stat['用户非工资收入次数']
print(user_bank_stat)
# user_bank_stat.to_csv(resPath + '/每人收入支出分布.csv')
'----------------------------------------------------------------------------------------------------------------------------------------'
user_outcome_month = user_train[(user_train['交易类型'] == 1)].groupby(['用户标识', 'month'], as_index=False)['交易金额'].agg(
    {'收入金额': 'sum'})
user_income_month = user_train[(user_train['交易类型'] == 0)].groupby(['用户标识', 'month'], as_index=False)['交易金额'].agg(
    {'支出金额': 'sum'})

print('按月统计 ... ')
TransferInMonth = user_bank_test[(user_bank_test['month'] == 1)].groupby(['用户标识'], as_index=False)['交易金额'].agg(
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
    tmp = user_train[(user_train['交易类型'] == 1) & (user_train['month'] == i)].groupby(['用户标识'], as_index=False)[
        '交易金额'].agg({str(i) + '月' + '支出金额': 'sum'})
    user_O_month = pd.merge(tmp, user_O_month, on='用户标识', how='inner')  # 支出

    tmp = user_train[(user_train['交易类型'] == 0) & (user_train['month'] == i)].groupby(['用户标识'], as_index=False)[
        '交易金额'].agg({str(i) + '月' + '收入金额': 'sum'})
    user_I_month = pd.merge(tmp, user_I_month, on='用户标识', how='inner')  # 收入

    tmp = user_bank_test[(user_bank_test['month'] == i)].groupby(['用户标识'], as_index=False)['交易金额'].agg(
        {str(i) + '月交易金额': 'sum'})
    TransferInMonth = pd.merge(TransferInMonth, tmp, on='用户标识', how='inner')  # 交易

    tmp = user_train[(user_train['工资收入标记'] == 1) & (user_train['month'] == i)].groupby('用户标识', as_index=False)[
        '交易金额'].agg({'用户工资收入-' + str(i) + '月': 'sum'})
    user_salary_month = pd.merge(user_salary_month, tmp, how='inner', on='用户标识')

    tmp = user_train[(user_train['工资收入标记'] == 0) & (user_train['month'] == i)].groupby('用户标识', as_index=False)[
        '交易金额'].agg({'用户非工资收入-' + str(i) + '月': 'sum'})
    user_non_salary_month = pd.merge(user_non_salary_month, tmp, how='inner', on='用户标识')
testBank = pd.merge(user_I_month, user_O_month, how='inner', on='用户标识')
testBank = pd.merge(user_non_salary_month, testBank, how='outer', on='用户标识')
testBank = pd.merge(user_salary_month, testBank, how='outer', on='用户标识')
testBank = pd.merge(TransferInMonth, testBank, how='outer', on='用户标识')
testBank = pd.merge(user_bank_stat, testBank, how='outer', on='用户标识')
testBank.fillna(0, inplace=True)  # 最终的银行分析表
credit_bank = pd.merge(testBank, user_credit_ult, how='outer', on='用户标识')
credit_bank.fillna(0, inplace=True)
print(credit_bank.columns.tolist())
print(credit_bank.shape)
print(credit_bank)
credit_bank.to_csv('./用户银行信用卡分析_test.csv')
