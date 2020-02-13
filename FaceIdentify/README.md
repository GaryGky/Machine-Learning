**一、**   **实验背景**

人脸关键点检测是人脸识别和分析领域中的关键一步，它是诸如自动人脸识别、表情分析、三维人脸重建及三维动画等其它人脸相关问题的前提和突破口。

把关键点的集合定义为形状(shape)，形状包含了关键点的位置信息，而这个位置信息一般可以用两种形式表示，第一种是关键点的位置相对于整张图像，第二种是关键点的位置相对于人脸框(标识出人脸在整个图像中的位置)。我们把第一种形状称作绝对形状，它的取值一般介于 0 到 w or h，第二种形状我们称作相对形状，它的取值一般介于 0 到 1。这两种形状可以通过人脸框来做转换。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

**二、**   **实验任务**

通过训练集给定的7500张图片和关键带你位置标签，尽量提高对测试集图片关键点坐标预测的准确性。

**三、**   **预先设想**

 

回归 -- 构造一个回归分析器，输入为图像和pts坐标文件，模型可以根据像素对应的坐标来进行学习，当输入了一个新的图片时，根据像素点确定坐标

`OpevCV`库对图片的读取、显示、保存都是通过`np.ndarray`实现的。

使用`cv2.imread()`函数来加载图片，该函数的形式为

```
cv2.imread(path, flags)
```

o   flag=0: 加载灰色图片

o   flag=1: 图片的透明度会被忽略，默认值是读取彩色

如果读取正确的话，会返回一个[height, width, channel]的`numpy`对象。height是图片高度，width是图片宽度，channel表示图片的通道。

图片的通道：灰度图【黑白图】是没有通道这一概念的，在彩色图中`opencv`读入的通道排列时`BGR`，而不是`RGB`，所以大概通道是与图像颜色相关的一个东西。

·       图片的显示可以使用`imshow()`, 也可以使用`matplotlib`函数工具包来显示

```
cv2.imshow('image',img) # 注意这里需要添加一个窗口名称



plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

plt.xticks([]), plt.yticks([]) # 隐藏x、y轴

plt.show()
```

`openCV`是以`BGR`(三原色)的模式读取彩色图像的，而`matplotlib`是以`RBG`的方式输出图片的，所以mat不能输出`openCV`读入的彩色图像。

·       查看图片信息

```
print(img.shape) # (high,wide,channel)

print(img.size) # 像素总数目

print(img.dtype)
```

图片矩阵的变换

```
#注意到，opencv读入的图片的彩色图是一个channel last的三维矩阵（h,w,c），即（高度，宽度，通道）

#有时候在深度学习中用到的的图片矩阵形式可能是channel first，那我们可以这样转一下

print(img.shape)

img = img.transpose(2,0,1)

print(img.shape)
```

图片的归一化处理

```
#因为opencv读入的图片矩阵数值是0到255，有时我们需要对其进行归一化为0~1

img3 = cv2.imread('1.jpg')

img3 = img3.astype("float") / 255.0 #注意需要先转化数据类型为float

print(img3.dtype)

print(img3)
```

**四、**   **编程实现与调试过程**

**(一)** **建模步骤**

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)

l  **数据准备阶段**

**1.**  **将7500个图片转换成96\*96的矩阵。**

**2.**  **对矩阵进行标准化处理**

**3.**  **对标签进行标准化处理，以更快地达到收敛。**

l  **数据集的划分：**

由于助教提供的数据已经划分了训练集和测试集，在训练集的基础之上，使用K折交叉验证的方法将7500条数据按照4:1的比例划分为训练集和验证集。

l  **配置模型**

在本次实验中，我主要尝试了利用深度卷积神经网络的方法对图片的特征进行提取，并且预测人脸的关键点坐标。与分类方法不同，分类方法对输出层需要添加softmax函数来提取类别表示概率最高的结果作为图片的标签，而在人脸关键点的回归问题中，我组将损失函数修改为![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

直接利用MSE作为损失函数，按照最小二乘法的方法进行模型学习。

l  **训练模型**

训练模型的主要方法是调整网络结构和训练参数，针对每种模型的参数变化带来的效果提升，在后文会详细列出。

**五、**   **神经网络模型分析**

**卷积神经网络**

l  CNN的特点：

局部感知：模仿人类对外界的认知过程：从局部到全局，片面到全面。在实现中，图像也是局部像素联系紧密的，如果每次将所有像素都输入模型，会导致效率低下且预测准确度不高。因此，不同于经典的全连接模式的神经网络模型，CNN使用局部相连的卷积神经网络将图像分块连接，大大减少了模型的参数。

参数共享：每张自然图像（人物、山水、建筑等）都有其固有特性，也就是说，图像其中一部分的统计特性与其它部分是接近的。这也意味着这一部分学习的特征也能用在另一部分上，能使用同样的学习特征。因此，在局部连接中隐藏层的每一个神经元连接的局部图像的权值参数（例如5×5），将这些权值参数共享给其它剩下的神经元使用，那么此时不管隐藏层有多少个神经元，需要训练的参数就是这个局部图像的权限参数（例如5×5），也就是卷积核的大小，这样大大减少了训练参数。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)

池化：随着模型网络不断加深，卷积核越来越多，要训练的参数还是很多，而且直接拿卷积核提取的特征直接训练也容易出现过拟合的现象。因此，为了降低过拟合风险，可以对不同位置区域提取出有代表性的特征（进行聚合统计，例如最大值、平均值等），这种聚合的操作就叫做池化，通常也称为特征降维 --- 有效地解决了决策树模型中出现的维度灾难。

**神经网络结构：**

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg)**（中间有部分省略）**

 

**对应的实现：**

l  **卷积层+池化层**

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image018.jpg)

l  **全连接层+输出层**

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image020.jpg)

l  **优化器和损失函数定义**

​             尝试过adam和sgd两种优化器：

\1.  Adam训练效率高，收敛速度快。它是基于梯度下降，而与SGD又存在显著不同，其核心区别在于计算更新步长时，增加了分母：梯度平方累积和的平方根。此项能够累积各个参数的历史梯度平方，频繁更新的梯度，则累积的分母项逐渐偏大，那么更新的步长相对就会变小，而稀疏的梯度，则导致累积的分母项中对应值比较小，那么更新的步长则相对比较大。

\2.  SGD收敛效率较慢。

（一个直观的理解是：梯度下降法好比一个人站在山顶，每次寻找离自己最近（梯度下降），在自己**步长范围**（学习率）内下降最快的点移动）

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image022.jpg)

在实验中adma表现效果优于SGD。主要体现在收敛速度上和预测的准确度。

​    **训练过程截图**

 ![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image024.png)

​    **参数调优**

n  调整batchSize：batch size,从128左右开始调整.batch size合适最重要,并不是越大越好。我在本实验中将batchSize从128调整到512。训练过程及其缓慢。

n  调整shuffle：每次选择交叉验证的比例。

**更多可以优化的地方**

\1.  使用深度神经网络进行多任务学习：

在阅读论文的过程中发现：MTCNN模型通过多任务学习的方式（每次训练三个神经网络，每个神经网络同时进行人脸识别，人脸对齐和关键点回归三个任务，预测出来的效果十分优秀）。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image026.jpg)

\2.  在另一篇论文中看到了通过heatmap对图像进行辅助预测，构建一张人脸边界图作为中间数据。因为人脸的关键点分布都是大部分都是基于边界分布的，所以在确定了边界之后，关键点回归的准确性将大幅提高。

同时，在该文章中提出的框架下，还能有效的处理对人脸面部遮挡，倾斜等噪声，提高预测模型的鲁棒性。因为在该实验中，我发现简单的神经网络模型甚至是深度神经网络模型已经无法适应人的侧脸和有遮挡的情况，要么预测结果准确率极低，要么直接无法预测出来，所以寒假准备利用课余时间进一步完善该项目。

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image028.jpg)

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image030.jpg)

**六、**   **遇到的问题**

1、          在测试集中，存在部分人脸是有遮挡的，如果模型的鲁棒性较差，无法处理该种样本。

2、          使用到openCV和cv2与pycharm版本不兼容

解决方法：使用Anaconda自带的编辑器Spyder，优点：方便管理python第三方包。

3、          深度学习框架caffe与windows版本不兼容的问题：只能使用队友电脑上的Linux系统运行。

**七、**   **参考文献**

【1】  大话神经网络经典模型LeTeX

 （<https://my.oschina.net/u/876354/blog/1632862>）

【2】  关于Lenet神经网络的认识：

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haner 

《*Gradient-Based Learning Applied to Document Recognition*》

【3】  基于多任务级联卷积神经网络的人脸检测和对齐《Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks》

【4】  Look at Boundary: A boundary-Aware Face Alignment Algorithm

**九、**   **排名截图**

**第九名**

![img](file:///C:/Users/dell/AppData/Local/Temp/msohtmlclip1/01/clip_image032.png)