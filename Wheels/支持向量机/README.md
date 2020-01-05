## 支持向量机

#### SVM简介

支持向量机是一种二分类模型，它的基本模型是定义在线性空间上的间隔最大的线性分类器，间隔最大使其区别于感知机模型。通过核函数（有时也称为：``Kernal Trick``）可以将线性空间扩展到高维空间上。其学习策略是**间隔最大化**，可以用求解凸二次规划问题来解决，也等价于正则化的合页损失函数的最小化问题。

![1574231917320](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1574231917320.png)

> 在本案例中，输入向量维度为12: **X**(x1,x2,x3, ... ,x12)，是一个求解高维度的超平面问题，所以以下着重讲述解决高维问题的算法流程。

#### 算法流程

![1574233981692](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1574233981692.png)

> KKT第三个条件称为KKT对偶互补条件。

#### SMO求解$\alpha$ 过程

- 选取一堆需要更新的变量 $$\alpha_i \space and \space \alpha_j$$
- 固定$$\alpha_i \space and \space \alpha_j$$ 以外的参数，求解拉格朗日对偶问题获得更新后的$$\alpha_i \space and \space \alpha_j$$

**注意：**直观理解，变量于KKT条件违背的程度越大，则变量更新后带来的目标函数增幅越大。因此SMO采用了一个启发式：使选取变量所对应的样本间隔最大。

#### 通过 $\alpha$ 求解W和b

![clip_image068[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201103/201103182044356550.png)

> 返回b中的 k 表示的是支持向量的下标集合。

#### 代码实现

- 数据分割

  1. 在线下实验的时候，将输入的数据集通过CV的方法按照5：1划分为训练集和测试集。

  2. 在训练的时候将train分出5000条数据作为包外验证。

  ```
  import pandas as pd
  import sklearn
  from pandas import DataFrame
  from sklearn.model_selection import train_test_split
  
  data = pd.read_csv("./svm_training_set.csv", index_col=False)
  
  y_labels = DataFrame(data[['index', 'label']])
  X_data = data.drop(['label'], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.20, random_state=33)  # 随机划分
  
  train = pd.merge(X_train, y_train, on='index')
  test = pd.merge(X_test, y_train, on='index')
  
  train.to_csv("train.csv", index=None)
  test.to_csv("test.csv", index=None)
  
  valid = train.iloc[0:5000, :]  # 构造验证集
  train = train.iloc[5000:, :]  # 构造训练集
  ```

  **注：如果输入数据已经准备好``train.csv``和``test.csv``，那么就跳过对数据集的划分，直接进入主函数**

- 数据标准化

  ```
  min_max_scaler = MinMaxScaler()
  train = pd.read_csv('train.csv', index_col=None)
  test = pd.read_csv('test.csv', index_col=None)
  ...
  dataArr = min_max_scaler.fit_transform(data_train.drop(['label', 'index'], axis=1))  # 读取训练数据 :: 进行标准化
  print('start SMO Process...')
  ```

  观察到``x3,x10,x11``的特征空间存在范围变化过大的情况，所以应该尝试剔除离群点或者标准化的方式来避免噪声。在本次实验中，我尝试使用了MinMaxScaler方法对数据进行标准化。

  利用MinMaxScaler标准化库将属性值缩放到一个指定的最大和最小值（通常是1-0）之间。

  使用这种方法可以：

  1、对于方差非常大的属性可以增强其稳定性。

  2、维持稀疏矩阵中为0的条目

- 由于训练集有上万条数据，训练耗时很大，故通过建立类，来保存中间数据

  ```
  # 定义类，方便存储数据
  class svmDataCached:
      def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
          self.X = dataMatIn  # 样本特征
          self.labelMat = classLabels  # 样本标签
          self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
          self.tol = toler  # 停止阀值
          self.m = shape(dataMatIn)[0]  # 样本数量
          self.alphas = mat(zeros((self.m, 1)))  # alpha --- 计算分界面的重要参数
          self.b = 0  # 初始设为0 --- 超平面到远点的距离
          self.eCache = mat(zeros((self.m, 2)))  # 缓存
          self.K = mat(zeros((self.m, self.m)))  # 核函数的计算结果
          for i in range(self.m):
              self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
  ```

- 定义一些中间重复利用的小函数

  ```
  verbose = 0 # 用于控制打印信息
  
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
  
  ```
  
- 最后是SMO的算法流程

  ```
  # SMO函数，用于快速求解出alpha
  # 输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
  def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
      feature = mat(dataMatIn)  # 输入训练维度
      label = mat(classLabels).transpose()  # 输入训练集标签
      # print(feature)
      # print(label)
      svm = svmDataCached(feature, label, C, toler, kTup)
      iter = 0
      entireSet = True
      alphaPairsChanged = 0
      while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
          print("iteration number: %d" % iter)
          alphaPairsChanged = 0
          if entireSet:
              for i in range(svm.m):  # 遍历所有数据
                  alphaPairsChanged += checkKKT(i, svm)  # 计算第i个样本是否满足ktt条件
                  print("fullSet, iter: %d i:%d, pairs changed %d" % (
                      iter, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
              iter += 1
          else:
              nonBoundIs = nonzero((svm.alphas.A > 0) * (svm.alphas.A < C))[0]
              for i in nonBoundIs:  # 遍历非边界的数据
                  alphaPairsChanged += checkKKT(i, svm)
                  print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
              iter += 1
          if entireSet:
              entireSet = False
          elif (alphaPairsChanged == 0):
              entireSet = True
      return svm.b, svm.alphas
  ```

#### 关于评估参数

![1574404952638](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1574404952638.png)

在该实验中，采用召回率准确率和F值评估模型的预测准确率

- Tp(True Positive) : 正确， 表示正样本预测正确
- Fp(Flase Positive) : 错误，表示正样本预测错误
- Tn(True Negative) : 正确， 表示负样本预测正确
- Fn(False Negative) : 错误，表示负样本预测正确

> 存在一个问题，在实验中，我偶然选择了一组交叉验证的验证集（只有十条数据，并且恰好全为负样本），通过SVM的预测全部预测成负类。但是这样计算出来的F1_Score为0。实际上对应的准确率应该为100%。
>
> 所以就此，我认为如果使用F1_score作为评估参数可能导致一些偶然误差，所以建议使用对数损失函数或者均方损失误差等。

#### 感想

- 通常在数据挖掘比赛中都是直接调包调库来解决问题，但是通过这次深入SVM了解其底层原理之后，发现自己在这方面的知识远远不足。调包调库可以在短短一个小时之内学会，但是了解SVM的实现过程却需要三天以上。通过这几天的琢磨，我对支持向量机问题，甚至于机器学习都有了全新的认识，相信这能让我在以后的学习中获得巨大的buff。
- 期间最大的困难在于SMO的数学推导上，需要用到很多高等代数的变换和数学分析的求导，反复看了三四遍才明白作者想要做啥，但理解了之后不得不感叹算法之精妙。总之，是一次非常值得的SVM实践。