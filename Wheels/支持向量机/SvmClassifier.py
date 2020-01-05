from numpy import *
from sklearn.preprocessing import MinMaxScaler

verbose = 1


# 随机选取出alpha_i之外的另一个alpha
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 对更新后的alpha进行规范化
def clipAlpha(aj, High, Low):
    if High < aj:
        aj = High
    if Low > aj:
        aj = Low
    return aj


# 计算违背KKT准则的程度
def calcEk(svm, k):
    fXk = float(multiply(svm.alphas, svm.labelMat).T * svm.K[:, k] + svm.b)
    Ek = fXk - float(svm.labelMat[k])
    return Ek


def updateEk(svm, k):  # 更新svm数据
    Ek = calcEk(svm, k)
    svm.eCache[k] = [1, Ek]


def kernelTrans(X, A, kTup):  # 核函数，输入参数,X:支持向量的特征数；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    for j in range(m):
        deltaRow = X[j, :] - A
        K[j] = deltaRow * deltaRow.T
    K = exp(K / (-1 * kTup[1] ** 2))  # 返回生成的结果
    return K


# 定义类，方便存储数据
class svmDataCached:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  # 数据特征
        self.labelMat = classLabels  # 数据类别
        self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler  # 停止阀值
        self.m = shape(dataMatIn)[0]  # 数据行数 --- 样本数量
        self.alphas = mat(zeros((self.m, 1)))  # alpha --- 计算分界面的重要参数
        self.b = 0  # 初始设为0
        self.eCache = mat(zeros((self.m, 2)))  # 缓存
        self.K = mat(zeros((self.m, self.m)))  # 核函数的计算结果
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# 随机选取aj，并返回其E值
def selectAlphaJ(i, svm, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    svm.eCache[i] = [1, Ei]
    validEcacheList = nonzero(svm.eCache[:, 0].A)[0]  # 返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(svm, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, svm.m)
        Ej = calcEk(svm, j)
    return j, Ej


# 首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def checkKKT(i, svm):  # 输入参数i和所有参数数据
    Ei = calcEk(svm, i)  # 计算E值
    if ((svm.labelMat[i] * Ei < -svm.tol) and (svm.alphas[i] < svm.C)) or (
            (svm.labelMat[i] * Ei > svm.tol) and (svm.alphas[i] > 0)):  # 检验这行数据是否符合KKT条件
        j, Ej = selectAlphaJ(i, svm, Ei)  # 随机选取aj，并返回其E值
        alphaIold = svm.alphas[i].copy()
        alphaJold = svm.alphas[j].copy()
        if (svm.labelMat[i] != svm.labelMat[j]):
            L = max(0, svm.alphas[j] - svm.alphas[i])
            H = min(svm.C, svm.C + svm.alphas[j] - svm.alphas[i])
        else:
            L = max(0, svm.alphas[j] + svm.alphas[i] - svm.C)
            H = min(svm.C, svm.alphas[j] + svm.alphas[i])
        if L == H:
            if verbose: print("L==H")
            return 0
        eta = 2.0 * svm.K[i, j] - svm.K[i, i] - svm.K[j, j]
        if eta >= 0:
            if verbose: print("eta>=0")
            return 0
        svm.alphas[j] -= svm.labelMat[j] * (Ei - Ej) / eta
        svm.alphas[j] = clipAlpha(svm.alphas[j], H, L)
        updateEk(svm, j)
        if (abs(svm.alphas[j] - alphaJold) < svm.tol):  # alpha变化大小阀值
            if verbose: print("j not moving enough")
            return 0
        svm.alphas[i] += svm.labelMat[j] * svm.labelMat[i] * (alphaJold - svm.alphas[j])
        updateEk(svm, i)  # 更新数据
        #  计算b
        b1 = svm.b - Ei - svm.labelMat[i] * (svm.alphas[i] - alphaIold) * svm.K[i, i] - svm.labelMat[j] * (
                svm.alphas[j] - alphaJold) * svm.K[i, j]
        b2 = svm.b - Ej - svm.labelMat[i] * (svm.alphas[i] - alphaIold) * svm.K[i, j] - svm.labelMat[j] * (
                svm.alphas[j] - alphaJold) * svm.K[j, j]
        if (0 < svm.alphas[i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# SMO函数，用于快速求解出alpha
# 输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup):
    feature = mat(dataMatIn)  # 输入训练维度
    label = mat(classLabels).transpose()  # 输入训练集标签
    # print(feature)
    # print(label)
    svm = svmDataCached(feature, label, C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        if verbose: print("iteration number: %d" % iter)
        alphaPairsChanged = 0
        if entireSet:
            for i in range(svm.m):  # 遍历所有数据
                alphaPairsChanged += checkKKT(i, svm)  # 计算第i个样本是否满足ktt条件
                if verbose: print("fullSet, iter: %d i:%d, pairs changed %d" % (
                    iter, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((svm.alphas.A > 0) * (svm.alphas.A < C))[0]
            for i in nonBoundIs:  # 遍历非边界的数据
                alphaPairsChanged += checkKKT(i, svm)
                if verbose: print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return svm.b, svm.alphas


import pandas as pd

# 主程序
min_max_scaler = MinMaxScaler()
train = pd.read_csv('train.csv', index_col=None)
test = pd.read_csv('test.csv', index_col=None)
print(test.describe())
data_train = train
data_test = test
labelArr = data_train['label']
dataArr = min_max_scaler.fit_transform(data_train.drop(['label', 'index'], axis=1))  # 读取训练数据 :: 进行标准化
print('start SMO Process...')

# 输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数:使用径向基函数
b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3))  # 通过SMO算法得到b和alpha
# 已经计算出来w和b
datMat = mat(dataArr)
labelMat = mat(labelArr).transpose()
svInd = nonzero(alphas)[0]  # 选取不为0数据的行数（也就是支持向量）
sVs = datMat[svInd]  # 支持向量的特征数据
labelSV = labelMat[svInd]  # 支持向量的类别（1或-1）
if verbose: print("there are %d Support Vectors" % shape(sVs)[0])  # 打印出共有多少的支持向量
""" 判断支持向量的划分正确性
m, n = shape(datMat)  # 训练数据的行列数
errorCount = 0
for i in range(m):
    kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))  # 将支持向量转化为核函数
    predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 这一行的预测结果注意最后确定的分离平面只有那些支持向量决定。
    if sign(predict) != sign(labelArr[i]):  # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
        errorCount += 1
print("the training error rate is: %f" % (float(errorCount) / m))  # 打印出错误率
"""

""" -------------- 测试部分 ---------------"""
labelArr_test = data_test['label']  # 测试集的标签
dataArr_test = min_max_scaler.fit_transform(data_test.drop(['label', 'index'], axis=1))  # 测试集的特征
errorCount_test = 0
datMat_test = mat(dataArr_test)
labelMat = mat(labelArr_test).transpose()
m, n = shape(datMat_test)
predict = []
for i in range(m):  # 在测试数据上检验错误率
    kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
    res = sign(kernelEval.T * multiply(labelSV, alphas[svInd]) + b)
    predict.append(res[0, 0])  # 得到输入向量的标签
print(predict)  # 计算结果
"""--------------线下得分验证-----------------"""
Tp = 0  # 正类预测称正类
Fp = 0  # 正类预测成负类
Tn = 0  # 负类预测成负类
Fn = 0  # 负类预测成正类
for i in range(m):  # 在测试数据上检验错误率
    kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
    predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 得到输入向量的标签
    if sign(predict) == 1 and sign(labelArr_test[i]) == 1:
        Tp += 1
    elif sign(predict) == -1 and sign(labelArr_test[i]) == 1:
        Fp += 1
    elif sign(predict) == -1 and sign(labelArr_test[i]) == -1:
        Tn += 1
    elif sign(predict) == 1 and sign(labelArr_test[i]) == -1:
        Fn += 1

print(Tp, Tn, Fn, Fp)
Pre = Tp / (Tp + Fp)
Rec = Tp / (Tp + Fn)
F1 = 2 * Pre * Rec / (Pre + Rec)
print("F1 Score is : ", F1)
