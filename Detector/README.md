

## CNN模型的可解释性

### CAM

> 一种saliency方法，是一种类似attention，gaze注意力集中机制的方法。计算机在处理视觉信息的时候需要【一个智慧的机制来滤除视觉数的中的错误的数据】。

#### saliency map

表示场景突出性的地形图，它的提出者【Koch 和 Ullman】引入了一个winner-take-all神经网络，这个网络会选择最显著的位置，利用返回一直机制使注意力焦点移向下一个最显著的位置。

##### 分类标准

- bottom-up（激励驱动）基于视觉**场景的特性**
- top-down(任务驱动) : 由认知现象如：知识、期望、奖励和当前任务决定。

##### bottom-up举例

> 吸引我们注意力的感兴趣区域必须充分地不同于其周围特征，这种注意力机制是外在的，自动的，灵活的**周边因素**。

```
一副在很多垂直条纹中只有一条水平条纹的场景图中，bottom-up注意力马上会注意到这条水平条纹。
```

##### top-down举例

> 由认知现象的知识、期望、奖励和当前任务决定。
>
> 它比较慢，任务驱动。
>
> 他展示了依靠当前任务的眼球运动的如下实验：测试者要求在不同的条件（问题）下看同一场景（在有一家人的房屋中，一个不速之客进入房间），这些问题包括：“估计这个家庭的物质环境”，“人们的年龄是多少”，或者简单的仔细观察场景。

1. 物体特征
2. 场景特征
3. 任务需求

### 数据集信息

CIFAR-10包含6W张图片的数据集，其中每张照片为32*32的彩色照片，每个像素包括RGB三个数值，范围(0,255)。

> 所有照片分属10个不同的类别，分别是 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
>
> 其中五万张图片被划分为训练集，剩下的一万张图片属于测试集。

#### 数据结构

```
dict_keys(['batch_label', 'labels', 'data', 'filenames'])
```

- labels：标记图片所属的标签，取值范围是0-9，代表当前图片的所属类别
- data：10000*3072（32*32*3）的二维数组，每一行代表一张图片的像素值
- filenames: 长度为10000的列表，每一项代表图片文件名。

### 原理图

![CNN_GAP](http://lc-cf2bfs1v.cn-n1.lcfile.com/e4a636f667f277cedc74.png)

引入GAP更充分的利用了空间信息，且没有了全连接层的各种参数，鲁棒性提高了，过拟合风险降低了。

![GAP_output](http://lc-cf2bfs1v.cn-n1.lcfile.com/2173d395c485a58a4b66.png)

经过GAP后，可以得到最后一个卷积层每个特征图的特征均值，通过加权和得到输出。需要注意的是，对每一个类别C，每个特征图K的均值都有一个对应的w，记为$$w_k^c$$。

在普通CNN训练完成之后，我们需要取得用于解释分类结果的热力图，即：将羊驼这个类别对应的所有$$W_k^c$$取出来，求它们与自己对应特征图的加权和即可。

### 算法流程

> 引用：对一个深层的卷积神经网络而言，通过多次卷积和池化以后，它的最后一层卷积层包含了最丰富的空间和语义信息，再往下就是全连接层和`softmax`层了，其中所包含的信息都是人类难以理解的，很难以可视化的方式展示出来。所以说，要让卷积神经网络的对其分类结果给出一个合理解释，必须要充分利用好最后一个卷积层。

#### 加载图片

1. 原始图片
2. 升维
3. 使用库进行预处理

#### 从输出结果中得到神经网络计算结果的解释信息

- 提取输出层的信息

- L2正则化

- 求出$$W_k^i$$每一个特征图对应最终标签的权重。

  方法是改变模型最后一层的连接关系。首先要得到最后一层卷积层的输出，然后使用GAP层得出特征权重，但是这种方法显然改变了模型的结构，不适用于网上训练、封装好的开源模型。

  比如在实验中就遇到了很棘手的问题：

  ![1575720108163](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1575720108163.png)

  这也是CAM局限之处，所以，我尝试实现了`grad-CAM`, 可以不用改变现有模型的结构。

### `grad-CAM`

采用全局梯度平均的方法代替了CAM中的GAP层计算每个特征图对于预测标签的权重。

> ~~事实上，经过严格的数学推导，Grad-CAM与CAM计算出来的权重是等价的。~~

为了和CAM中的权重区分，定义Grad-CAM中第k个特征图对预测类别C的权重为：
$$
\alpha_k^c = \frac{1}{Z}\sum_i\sum_j\frac{\partial{y^c}}{\part{A^k_{ij}}}
\\
Z为特征图像素个数 
\\
y^c是对应类别c的分数（在代码中一般使用logits表示，是输入softmax层之前的值
\\
A_{ij}^k表示在第k个特征图中(i,j)位置的像素值。
$$

#### 对应的pipeline为

![Grad-CAM结构](http://lc-cf2bfs1v.cn-n1.lcfile.com/531c1d38e0a0ddaf9553.jpg)

grad-CAM对于最终的加权和加了一个ReLu，是为了剔除一些对类别c没有影响的点；如果不加ReLu，模型可能会带入一些其他类别的像素，从而影响解释的效果。

#### grad-CAM + 热力图 得出的网络模型解释

![1575722120894](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1575722120894.png)

#### grad-CAM + 导向反传播相结合得到更细致的解释。

![effect2](http://lc-cf2bfs1v.cn-n1.lcfile.com/9c1076aee75f1d4a98ef.png)

### 一些函数

#### `np.squezz`

用于图像降维

#### `keras.Lambda层`

> 如果你只是想对流经该层的数据做个变换，而这个变换本身没有什么需要学习的参数，那么直接用Lambda Layer是最合适的了。
>
> 在CAM中，我们只需要对全连接层前的最后一层卷积层的输出进行数据变换，然后加以权重，就可以得到解释信息。

用于对上一层的输出施加`tensorflow`表达式。

接受两个参数：输入张量对输出张量的映射函数，输入shape对输出shape的映射函数。

#### `关于keras.Model函数`

**与顺序模型Sequence相称的另一个函数式API**

- 函数式模型接口

  `Keras`的函数式模型为`Model`，即广义的拥有输入和输出的模型，我们使用`Model`来初始化一个函数式模型

- 常用的模型属性

  - `model.layers`：组成模型图的各个层
  - `model.inputs`：模型的输入张量列表
  - `model.outputs`：模型的输出张量列表

- Model模型方法

  - compile

    ```
    compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    ```

    > 【Tips】如果只是载入模型并利用其predict，可以不用进行compile。在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。predict会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数)

  - fit

    ```
    fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    ```

    `fit`函数返回一个`History`的对象，其`History.history`属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况

  - 更多的父类`API`

    - `model.get_weights()` 返回模型中所有权重张量的列表，类型为 Numpy 数组。
    - `model.set_weights(weights)` 从 Numpy 数组中为模型设置权重。列表中的数组必须与 `get_weights()` 返回的权重具有相同的尺寸。
    - `model.to_json()` 以 JSON 字符串的形式返回模型的表示。请注意，该表示不包括权重，仅包含结构。你可以通过以下方式从 JSON 字符串重新实例化同一模型（使用重新初始化的权重）：
    - etc...

### L2正则化

在深度学习中，用的比较多的正则化技术是L2正则化，其形式是在原先的损失函数后边再加多一项：$$\frac{1}{2}λθ^2_i\frac{1}{2}λθ_i^2$$，那加上L2正则项的损失函数就可以表示为：$$L(θ)=L(θ)+λ\sum^{n}_{i}θ^2_i$$，其中θ就是网络层的待学习的参数，λ则控制正则项的大小，较大的取值将较大程度约束模型复杂度，反之亦然。

L2约束通常对稀疏的有尖峰的权重向量施加大的惩罚，而偏好于均匀的参数。这样的效果是鼓励神经单元利用上层的所有输入，而不是部分输入。所以L2正则项加入之后，权重的绝对值大小就会整体倾向于减少，尤其不会出现特别大的值（比如噪声），即网络偏向于学习比较小的权重。所以L2正则化在深度学习中还有个名字叫做“权重衰减”（weight decay），也有一种理解这种衰减是对权值的一种惩罚，所以有些书里把L2正则化的这一项叫做惩罚项（penalty）。

我们通过一个例子形象理解一下L2正则化的作用，考虑一个只有两个参数w1w1和w2w2的模型，其损失函数曲面如下图所示。从a可以看出，最小值所在是一条线，整个曲面看起来就像是一个山脊。那么这样的山脊曲面就会对应无数个参数组合，单纯使用梯度下降法难以得到确定解。但是这样的目标函数若加上一项0.1×(w21+w22)0.1×(w12+w22)，则曲面就会变成b图的曲面，最小值所在的位置就会从一条山岭变成一个山谷了,此时我们搜索该目标函数的最小值就比先前容易了，所以L2正则化在机器学习中也叫做“岭回归”（ridge regression）。

### Keras后端

#### function

实例化一个Kears函数

```
function(inputs, outputs, updates=[])
input : 列表 of 占位符或者张量变量
output : 输出张量列表
```

#### l2_normalize

在给定轴上对张量进行L2范数规范化

#### image_data_format

#### set_image_data_format

查看和更改图像的通道顺序【与颜色RGB有关】。

#### `eval`

求得张量的值，返回一个Numpy array

```
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
   [ 3.,  4.]], dtype=float32)
```

#### `zeros_like`

生成与另一个张量x的shape相同的全0张量。

#### 还可以有

2D池化，3D池化等方法。

### `TensorFlow`

- #### gradient(`ys`,`xs`)

  计算ys对于xs方向上的梯度

- #### stop_gradients()

  阻挡结点BP的梯度，一个结点被stop之后就无法继续向前BP了。

### 关于激活函数ReLu（修正线性单元）

神经元卷积层常用的激活函数。

![image.png](https://bbs-img.huaweicloud.com/blogs/img/1565666532482213.png)



### 拓展

### Q：

1. 彩色图像的通道的概念应该怎么理解？
2. 封装成类。
3. 训练模型。
4. `url接口`

### 主要参考了这篇文章

[URL: 凭什么相信你的CNN模型](https://bindog.github.io/blog/2018/02/10/model-explanation/#%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99)