# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:26:19 2019

@author: dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Saliency():
    '''
    这里定义了计算saliency maps时的通过函数
    实现具体的method时，主要是注册hook
    之后的计算由父类Saliency的函数实现
    '''
    def __init__(self, model): # model为训练好的model : 我的使用cifar-10
        self.model = model 
        self.handles = [] # 句柄
        self.saliency_maps = [] # 在这里是heatmap和cam
        self.grad_blocks = [] # 每张图对于类别C的解释信息 - 即：梯度
        self.model_output = None # 模型的输出层
        self.model_pred = None # 模型对于输入的预测结果

    def get_datagrad(self, x): # 计算解释信息
        '''
        对loss backward()计算梯度
        :param x: 输入图片
        :return: saliency maps
        '''
        self.model_output = self.model(x) # 直接输入模型得到输出信息
        self.model.zero_grad() # 
        
        self.model_pred = self.model_output.max(1, keepdim=True)[1].squeeze(1) # 预测标签一个softmax层
        loss = F.cross_entropy(self.model_output, self.model_pred) # Loss计算用的标签是模型输出标签
        loss.backward(retain_graph=True) # torch中计算导数的方法
        self.data_grad = x.grad.data #图片的梯度信息
        self.saliency_maps = torch.cat([x, self.data_grad], 1)

        return self.saliency_maps

    def release(self):
        '''
        释放hook和内存，每次计算saliency后都要调用release()
        :return:
        '''
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.saliency_maps = []
        self.grad_blocks = []
        self.model_output = None
        self.model_pred = None
        self.data_grad = None
