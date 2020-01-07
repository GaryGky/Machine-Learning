import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

trainpath = "../data/train"
testpath = "../data/A"
resPath = trainpath + '/result'

bank = pd.read_csv(resPath + '/银行卡流水时间为0.csv')
print(bank.drop('Unnamed: 0',inplace=True,axis=1))
credit = pd.read_csv(resPath + '/信用卡时间戳为0.csv')
print(credit.drop('Unnamed: 0',inplace=True,axis=1))

merge = pd.merge(bank,credit,how='inner',on='用户标识')
print(merge[(merge['用户标识'] == 347)])
