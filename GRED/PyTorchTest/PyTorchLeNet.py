# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:09:43 2019

@author: dell
"""

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
    
    
