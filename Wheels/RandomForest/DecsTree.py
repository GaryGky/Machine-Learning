# 定义一棵决策树
class Tree(object):
    def __init__(self):
        self.split_feature = None  # 当前结点用来划分的属性
        self.split_value = None  # # 划分结点的阈值
        self.leaf_value = None  # 叶结点的值
        self.tree_left = None  # 左边是一颗决策树
        self.tree_right = None  # 右边是一颗递归决策树

    # 通过递归决策树找到样本所属叶子节点
    def calc_predict_value(self, dataset):  # 训练的时候使用
        if self.leaf_value is not None:  # 递归终止条件 :: 只有叶子结点有值
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:  # 如果小于阈值，丢到左边
            return self.tree_left.calc_predict_value(dataset)
        else:  # 不然丢到右边
            return self.tree_right.calc_predict_value(dataset)

    # 以json形式打印决策树，方便查看树结构
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:  # 如果没有左右子树，说明是叶节点，打印叶节点
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()  # 递归调用，打印左子树
        right_info = self.tree_right.describe_tree()  # 递归调用，打印右子树
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

