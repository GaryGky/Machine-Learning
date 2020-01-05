import pandas as pd
import numpy as np
import random
import math
import collections

from RandomForest.DecsTree import Tree
debug = 0

# 随机森林
class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree="sqrt", subsample=1.0, random_state=None):
        # 定义分类器的属性
        self.n_estimators = n_estimators  # 迭代次数
        self.max_depth = max_depth if max_depth != -1 else float('inf')  # 决策树的最大深度:如果是-1，就不限制树深度，否则设置为无穷（）
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf  # 最小叶节点数量
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree  # 列采样
        self.subsample = subsample  # 行采样
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances_ = dict()  # 映射关系 : {name: importance} 属性名的重要程度

    def fit(self, dataset, targets):  # 对训练集进行拟合
        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)  # 随机采样

        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5) * 2 # 选择二次根下的属性数量作为分类标准
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:  # 全部采样
            self.colsample_bytree = len(dataset.columns)

        for stage in range(self.n_estimators):
            print(("iter: " + str(stage + 1)).center(80, '='))
            # bagging方式随机选择样本和特征
            random.seed(random_state_stages[stage])
            subset_index = random.sample(range(len(dataset)), int(self.subsample * len(dataset)))
            subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
            dataset_copy = dataset.loc[subset_index, subcol_index].reset_index(drop=True)  # 获取特征
            targets_copy = targets.loc[subset_index, :].reset_index(drop=True)  # 获取标签

            tree = self.buildDecsTree(dataset_copy, targets_copy, depth=0)  # 递归建立决策树
            self.trees[stage] = tree  # 为森林新加入一棵树
            print(tree.describe_tree())  # 打印决策树信息

    # 递归建立决策树 :: 递归终止条件 :: 纯度一样
    def buildDecsTree(self, dataset, targets, depth):
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            # 该节点分裂使用的特征，阈值以及信息增益
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()  # 初始化一颗决策树
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self.buildDecsTree(left_dataset, left_targets, depth + 1)
                tree.tree_right = self.buildDecsTree(right_dataset, right_targets, depth + 1)
                return tree
        else:  # 如果树深度超过设定值，则直接终止分裂
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    # 选择最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益
    def choose_best_feature(self, dataset, targets):
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            else:  # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:  # 更新条件：大于当前值
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    # 选择样本中出现次数最多的类别作为叶子节点取值
    @staticmethod
    def calc_leaf_value(targets):
        label_counts = collections.Counter(targets)  # 计算label的数量
        if debug : print(label_counts)  # 输出应为: -1: num1; 1: num2
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]  # 以叶子结点中类别较多的标签作为总体标签

    # 计算基尼指数作为决策树分裂的特征选择标准
    @staticmethod
    def calc_gini(left_targets, right_targets):
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            label_counts = collections.Counter(targets)  # 统计每个类别有多少样本，然后计算gini
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    # 根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值
    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    # 输入样本，预测所属类别
    def predict(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for stage, tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row)) # 使用森立中每棵树预测一遍

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys())) # 投票法选出预测标签
            res.append(pred_label[1])
        return np.array(res)