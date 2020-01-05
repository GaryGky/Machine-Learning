## 随机森林

### 理论背景

#### 集成学习

通过将多个学习器进行集成，常可获得比单一学习器显著优越的泛化性能。这对弱分类器尤为明显。

- 弱分类器：准确率仅比随机猜测略高的分类器
- 强分类器：准确率高并能在多项式时间内完成的分类器

> 关键假设：基学习器的误差相互独立

有了上述假设，如果我们通过T个基分类器进行投票产生预测结果，若有超过半数的基分类器正确，我们便可以得到正确的结果。
$$
H(X) = sign(\sum_{i=1}^{T}h_i(x))
$$
由HOEFFDING不等式可知：集成的错误率为：

![1574318963367](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1574318963367.png)

**上式可以说明，随着集成基分类器的数量不断增大，错误率将成指数级下降，最终趋向于0。**

根据个体学习器的生成方式，目前的集成学习可以大致分为两类：

- 个体学习器之间存在强依赖关系、必须串行生成的序列化方法。（代表：Boosting）。
- 个体学习器之间不存在强依赖关系、可同时生成并行化方法。（代表：Bagging和随机森林）。

以下着重对第二类集成学习进行分析。

#### 随机森林

> 由于集成学习中的关键假设：个体学习器之间相互独立。这在现实生活或中往往不成立，因此Bagging采用了随机【有放回】的采样方式来产生采样集，用于训练不同的基学习器。

值得一提的是，自助采样过程带来了另一个优点：由于每个基学习器只使用了63.2%的训练集，剩余36.8%的训练集可以作为验证集来对泛化性能进行包外估计【类似Cross-Validation的思想】。

随机森林作为Bagging的一个扩展变体，在以决策树为基学习器集成Bagging的基础之上，在决策树的学习过程中引入了属性的随机选择。即：传统的决策树在结点分裂的时候需要对当前每个叶子结点（假设有d个）进行考察并从中挑选出一个属性进行结点分裂；而随机森林算法首先从中挑选k个属性，然后在这k个属性中选择1个最佳属性进行结点分裂。如果k=d，则退化为传统的决策树模型；如果k=1，则随机选择一个属性进行划分；通常情况下，推荐k = log(d).

##### 准确度考察

> 个体学习器的准确性和多样性存在冲突，集成学习的研究核心在于：产生“好而不同”的个体学习器。

随机森林通过样本扰动的方式对初始训练集采样而带来了显著不同，其多样性不仅来自于样本扰动，在决策树分裂的时候，会用到属性扰动，这使得模型最终的泛化性能得以提升。

##### 效率值考察

随机森林的效率优于Bagging，因为在个体决策树的构建过程中，Bagging使用的是确定型决策树，在划分属性的时候需要对每个结点的所有属性进行考察；而随机森林使用的随机性决策树只需要考察全部属性的一个子集。

#### 决策树

> 目标是生成一颗泛化能力强的分类器。
>
> 策略：分而治之。





### 问题分析

![1574318447098](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1574318447098.png)

> 二分类

### 数据处理

#### 自行划分训练集和测试集

```
train = pd.read_csv('data/x_train.csv', index_col=None)
test = pd.read_csv('data/y_train.csv', index_col=None)
data = pd.merge(train, test, on='index')
train_y = data[['index', 'label']]
X_train, X_test, y_train, y_test = train_test_split(data, train_y, test_size=0.2, random_state=0)
train = pd.merge(X_train, y_train, on=['index','label']) # 4548条数据
test = pd.merge(X_test, y_test, on=['index','label']) # 1138 条数据
```

#### 数据集特征

> 通过`df.describe()`发现，数据集在特征上的方差并不大，因此省去了归一化标准化处理。

![1575283167441](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1575283167441.png)

### 算法流程

1. 原始训练集为N，应用bootStrap法有放回地随机抽取k个新的自助样本集，并由此构造k棵分类树，每次违背抽到的样本组成了k个包外数据，用来进行包外估计。
2. 设有numFeature个特征，在每一棵树的节点处随机抽取numTry = log(numFeature)个变量，并按照Gini指数选出其中一个最佳分类变量。
3. 每棵树最大限度的生长，直到一个叶子内结点的纯度到达极限，并且不做任何剪枝处理。
4. 将生成的k课多分类基础树组成随机森林，利用随机森林分类器对测试集数据进行分类，分类结果按树分类器的投票数量决定。

### 实验过程（模型搭建/调参优化）

#### 随机森林分类器的参数说明

```
def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
             min_split_gain=0.0, colsample_bytree="sqrt", subsample=1.0, random_state=None):
    # 定义分类器的属性
    self.n_estimators = n_estimators # 对原始数据集进行有放回抽样生成的子数据集个数，即决策树的个数
    self.max_depth = max_depth if max_depth != -1 else float('inf') # 决策树的最大深度
    self.min_samples_split = min_samples_split # 决策树分裂的阈值
    self.min_samples_leaf = min_samples_leaf # 决策树上叶子的分裂阈值
    self.min_split_gain = min_split_gain # 阈值 : 决策树分裂的最小值增益
    self.colsample_bytree = colsample_bytree  # 列采样 :: 属性扰动
    self.subsample = subsample  # 行采样 :: 样本扰动
    self.random_state = random_state
    self.trees = dict()
    self.feature_importances_ = dict()
```

- n_estimators : 如果设置过大则容易过拟合【显然的，因为迭代次数越多，分类器对训练集的了解就越准确，当输入新的测试集，泛化效果可能很低】；如果设置过小，模型训练不充分，容易导致欠拟合。
- max_depth：如果设置为-1，就不限制决策树的最大深度【默认值是不限制的】。如果规定了分类器的最大树深度，那么就取输入值。该样本的特征不是很多，因此我选择了不设置树的最大深度
- min_sample_split：阈值：低于这个阈值，决策树停止分裂，独立成为森林中的一棵树。
- min_sample_leaf：阈值：低于这个值，一棵树上的叶子将停止分裂，并且使用该叶子上的样本中标签多数作为该叶子的标签。
- min_split_gain：决策树分裂的最小增益，一旦小于该值，决策树分裂也没有必要，因为信息增益不高，并且会加大过拟合的风险。
- random_state：如果等于某个固定值，那么每次划分将产生同样的数据集。如果设置为None，那么每次划分将得到不同的数据集。
- trees()：该随机森林中决策树的map（由序号映射到一棵树 ）
- feature_importance: 样本选择中特征的重要程度。

### 实验感想

- 原来决策树还能这么用？！在学习了集成学习之后，发现我们经常使用的决策树原来还可以组成一片森林。但是由于算法的限制，我们无法使用一些高级的决策树模型比如：XgBoost, LightGbm作为基础树，而是自己实现了一颗决策树轮子。有一种非常神奇的体验。

- 在coding的过程中，既担心自己捏出来的决策树肯定满是过拟合的风险，又不断的担心不做剪枝处理的分类，叶节点最终会不会分成一个结点一个样本的情况，好在训练结果似乎还不错。

- 之前调参总是需要到各大博客搜索【调参优化方法】，但自己亲手搭了一片森林之后，深刻理解了决策树参数的各种含义以及对训练的影响。个人认为这是从数据挖掘这门课上得到最有价值的内容。

- 在调库的时候偶然发现了一个sklearn的神器，GridSearcherCV(),它使用交叉验证的方式针对某一分类器指定想要的调参名称和数值，作为一个字典传入该函数，就会返回一个最佳的参数组合，使用案例如下：（这对我对调参过程起了很大帮助作用）

  ```
  ranForest = RandomForestClassifier()
  
  tuned_para_list = [{'n_estimators': [100, 200, 300, 400, 500],
                      'max_features': ['auto', 'sqrt', 'log2']}]
  
  clf = GridSearchCV(estimator=ranForest,
                     param_grid=tuned_para_list
                     , cv=5)
  clf.fit(train.drop(['index', 'label'], axis=1), train['label'])
  print('Best Parameters:')
  print(clf.best_params_)
  
  输出：
  Best Parameters:
  {'max_features': 'auto', 'n_estimators': 100}
  ```

- 附上随机森林【包】模型的调参实例，好奇官方包的结果如何做的。

  ```
  train = pd.read_csv('train.csv', index_col=False)
  test = pd.read_csv('test.csv', index_col=False)
  predict = DataFrame()
  predict['index'] = test['index']
  ranForest = RandomForestClassifier(n_estimators=100, max_features='auto')
  ranForest.fit(train.drop(['index', 'label'], axis=1), train['label'])
  predict['predict_label'] = ranForest.predict(test.drop(['index', 'label'], axis=1))
  predict = pd.merge(test[['index', 'label']], predict, on='index')
  print(predict)
  Tp = 0
  Fp = 0
  Tn = 0
  Fn = 0
  for index, row in predict.iterrows():
      # print(index)
      # print(row)
      if (row['label'] == 1 and row['predict_label'] == 1):
          Tp += 1
      elif row['label'] == -1 and row['predict_label'] == 1:
          Fp += 1
      elif row['label'] == -1 and row['predict_label'] == -1:
          Tn += 1
      elif row['label'] == 1 and row['predict_label'] == -1:
          Fn += 1
  Pre = Tp / (Tp + Fp)
  Rec = Tp / (Tp + Fn)
  F1 = 2 * Pre * Rec / (Pre + Rec)
  print(Tp, Tn, Fn, Fp)
  print("F1 score: ", F1)
  
  输出:
  TP:  555 Tn:  583 Fn:  0 Fp:  0
  F1 score:  1.0
  # 尽然预测全对…惊了。
  ```


### CAN WE DO BETTER

> 在训练集较小的情况下，可以通过设置max_depth和迭代次数来防止过拟合。但当轮子遇到了工业界的大数据，可能需要更完善的防止过拟合机制。

比如：剪枝。

### 参考资料

机器学习 --- 周志华 (西瓜书)