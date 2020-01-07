### PyTorch中的神经网络

#### Autograd自动求导

> autoGrad包提供Tensor所有操作的自动求导方法

- ##### `autograd.Variable`是这个包中最核心的类，它包装了一个Tensor并且支持所有在其上的操作，一旦完成计算后，可以通过`.backward()`来计算导数。

![img](https://pic4.zhimg.com/80/v2-08e0530dfd6879ff2bee56cfc5cc5073_hd.jpg)

- ##### `Function`

  Variable 和 Function 二者相互联系并且构建了一个描述整个运算过程的无环图。每个Variable拥有一个 .creator 属性，其引用了一个创建Variable的 Function。(除了用户创建的Variable其 creator 部分是 None)

  > 如果你想要进行求导计算，你可以在Variable上调用.backward()。 如果Variable是一个标量（例如它包含一个单元素数据），你无需对backward()指定任何参数，然而如果它有更多的元素，你需要指定一个和tensor的形状想匹配的grad_output参数。

  类似求偏导的过程！

  ~~一些例子,balabala...~~

  ```
  from torch.autograd import Variable
  x = Variable(torch.ones(2, 2), requires_grad = True)
  y = x + 2
  y.creator
  
  # y 是作为一个操作的结果创建的因此y有一个creator 
  z = y * y * 3
  out = z.mean()
  
  # 现在我们来使用反向传播
  out.backward()
  
  # out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
  # 在此处输出 d(out)/dx
  x.grad
  
  x = torch.randn(3)
  x = Variable(x, requires_grad = True)
  y = x * 2
  while y.data.norm() < 1000:
      y = y * 2
  gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
  y.backward(gradients)
  x.grad
  ```

#### PyTorch 搭建第一个神经网络

```
class LeNet(nn.Module): # 构建一个简单的神经网络
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5,120) # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, input):
        input = Func.max_pool2d(Func.relu(self.conv1(input)), (2,2)) # 池化
        input = Func.max_pool2d(Func.relu(self.conv2(input)), 2) # 池化
        input = input.view(-1,self.num_flat_features(input))
        input = Func.relu(self.fc1(input)) 
        input = Func.relu(self.fc2(input))
        input = self.fc3(input)
        return input
    
    def num_flat_features(self,x): # 计算图像特征数量的函数
        size = x.size()[1:] # all dimension except the batchs
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

> 由于PyTorch中自带backward函数，故只需定义一个前馈神经网络。
>
> 模型中可学习的参数可由net.parameters()返回

#### 添加输入观察输出

```
	input = Variable(torch.randn(1,1,32,32))
    output = leNet(input)
    print(input)
    print(output)
    
    leNet.zero_grad() # 对所有参数的梯度缓冲域进行归零
    output.backward(torch.randn(1,10)) # 使用随机梯度进行反传
```

> **注意: torch.nn 只接受小批量的数据**
> 整个torch.nn包只接受那种小批量样本的数据，而非单个样本。 例如，nn.Conv2d能够结构一个四维的TensornSamples x nChannels x Height x Width。
> *如果你拿的是单个样本，使用input.unsqueeze(0)来加一个假维度就可以了。*

#### 为网络添加损失函数

```
target = Variable(torch.range(1,10)) # [0,1,2,3,4,5,6,7,8,9]
loss_mse = nn.MSELoss(output,target)    
```

至此，整个网络的流程图如下：

```text
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d  
      -> view -> linear -> relu -> linear -> relu -> linear 
      -> MSELoss
      -> loss
```

#### 调用backward函数来对输入进行梯度求值

```
critertion = nn.MSELoss()
    loss_mse = critertion(output, target)
    leNet.zero_grad() # 对所有参数的梯度缓冲域进行归零
    loss_mse.backward() # 使用随机梯度进行反传
```

#### 定义权重更新（学习）

> 最简单的梯度下降法

```
learning_rate = 0.01
    for f in net.parameters:
        f.data.sub_(f.grad.data * learning_rate)
```

> 利用optim这个python的包定义优化器

```
import torch.optim as opt
    optimizer = opt.SGD(leNet.parameters(),lr = 0.01)
    optimizer.zero_grad() # 将梯度清零
    ...
    optimizer.step() # update weight
```

通常来说，在处理图像声音或者文本之前，需要将数据先转化为numpy中的数组，之后再转化为Tensor。

### 利用PyTorch读取CIFAR10

### python 面向对象

类的私有属性：两个下划线开头，声明该属性为私有，不能在类的外部使用或者直接访问。在类内部方法中使用时：`self.__privateAttr`

#### 关于下划线的说明

- `__Call__`: 定义的是特殊方法，一般是系统定义的名字，类似`__init__()`之类的。
- `_Func`：以单下划线开头的表示的是protected类型的变量，只允许本身及其子类访问。
- `__func`：双下划线表示私有类型的变量，只允许该类内部进行访问。

#### 特殊之处

在python类内部定义方法的时候，类方法必须包含一个`self`参数，并且为第一个参数。

> 感觉`self`应该是一个类似`this`指针的东西

#### 静态方法的定义

静态方法由类调用，与对象无关，因此不需要在方法的参数表中添加`self`关键词。只需要在方法的定义上方加上`@staticmethod`，就成为静态方法。

```
class ff:
    @staticmethod
    def runx():
        print("hello world!")
ff.runx()
```

#### 小实验

> 包括类定义和继承与多态的简单实现，主要是为了学习`python OOP`的语法

```
class Pet:
    def __init__(self,name,age,sex):
        self.name = name
        self.age = age
        self.sex = sex
    
    def printName(self):
        print("This is %s" % self.name)
    
    def selfIntroduction(self):
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))
    
class Cat(Pet):
    def selfIntroduction(self):
        print("I am a Cat!")
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))
class Dog(Pet):
    def selfIntroduction(self):
        print("I am a Dog!")
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))

cat = Cat("Tom",10,"Boy")
dog = Dog("Sponge",5,"Girl")

cat.selfIntroduction()
dog.selfIntroduction()
```

#### 附：使用PyTorch定义神经网络的完整代码

```
from torch.autograd import Variable
import torch

def TestGrad1():
    x = Variable(torch.ones(2,2),requires_grad = True)
    y = x+2
    z = y*y*3
    out = z.mean()
    print("out is: ", out)
    bw = out.backward()
    print("backward result is :", bw)
    x_grad = x.grad
    print("grad of x is: ", x_grad)

    
def TestGrad2():
    x = torch.randn(3)
    x = Variable(x,requires_grad = True)
    y = x*2
    while y.data.norm() < 1000:
        y *= 2
    # gradients = torch.FloatTensor([0.1,1.0,0.001])
    y.backward()
    x_grad = x.grad
    print("x_grad is ", x_grad)
    
import torch.nn as nn
import torch.nn.functional as Func

class LeNet(nn.Module): # 构建一个简单的神经网络
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5,120) # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, input):
        input = Func.max_pool2d(Func.relu(self.conv1(input)), (2,2)) # 池化
        input = Func.max_pool2d(Func.relu(self.conv2(input)), 2) # 池化
        input = input.view(-1,self.num_flat_features(input))
        input = Func.relu(self.fc1(input)) 
        input = Func.relu(self.fc2(input))
        input = self.fc3(input)
        return input
    
    def num_flat_features(self,x): # 计算图像特征数量的函数
        size = x.size()[1:] # all dimension except the batchs
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
        
if __name__ == '__main__' :
    leNet = LeNet()
    import torch.optim as opt
    optimizer = opt.SGD(leNet.parameters(),lr = 0.01)
    optimizer.zero_grad() # 将梯度清零
    # print(leNet)
    paras = list(leNet.parameters())
    # print(paras[0].size) # 第一层卷积层可以学习的参数
    # print(len(paras)) # 10:: 打出的是输出层的长度，
    # 因为最后一层output只有是个输出，分别表示cifar的10个类别
    input = Variable(torch.randn(1,1,32,32)) # 产生标准正态分布的随机数
    output = leNet(input)
    target = Variable(torch.range(1,10)) # [1,2,3,4,5,6,7,8,9,10]
    critertion = nn.MSELoss()
    loss_mse = critertion(output, target)
    leNet.zero_grad() # 对所有参数的梯度缓冲域进行归零
    loss_mse.backward() # 使用随机梯度进行反传
    print("网络的卷积层1的梯度：\n",leNet.conv1.bias.grad) # 求第一层卷积层的梯度
    print("网络的卷积层2的梯度：\n",leNet.conv2.bias.grad) # 求第一层卷积层的梯度
    
    # print("优化前的网络参数 : \n", list(leNet.parameters())[0]) # 优化前的参数
    learning_rate = 0.01
    for f in leNet.parameters():
        f.data.sub_(f.grad.data * learning_rate)
    # print("优化后的网络参数 : \n",list(leNet.parameters())[0])
    optimizer.step() # 更新权值
```

