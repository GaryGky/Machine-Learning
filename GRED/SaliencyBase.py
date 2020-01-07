import torch
import torch.nn as nn
import torch.nn.functional as F

class Saliency():
    '''
    这里定义了计算saliency maps时的通过函数
    实现具体的method时，主要是注册hook
    之后的计算由父类Saliency的函数实现
    '''
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.saliency_maps = []
        self.grad_blocks = []
        self.model_output = None
        self.model_pred = None

    def get_datagrad(self, x):
        '''
        对loss backward()计算梯度
        :param x: 输入图片
        :return: saliency maps
        '''
        self.model_output = self.model(x)
        self.model.zero_grad()
        
        self.model_pred = self.model_output.max(1, keepdim=True)[1].squeeze(1)
        loss = F.cross_entropy(self.model_output, self.model_pred) # Loss计算用的标签是模型输出标签
        loss.backward(retain_graph=True)
        self.data_grad = x.grad.data
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
