# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:26:44 2019

@author: dell
"""
import cv2
from skimage import img_as_float
from SaliencyBase import Saliency
import torch
import numpy as np
import torch.nn as nn
import torch.nn.Model as Model
from torch.autograd import Variable


class CamDetector(Saliency):
    def __init__(self, model):
        super(CamDetector, self).__init__(model)
        self.classNum = 10  # CIFAR10的类别数量
        self.forward_relu_output = []
        # self.update_relus()
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.sourceImg = ""  # 图片的输入路径 :: 可以是单张图片也可以是batch形式
        self.layer = ""  # 得到卷积神经网络的最后一层信息

    def deprocess_image(self, x):
        print("start deprocessing image...")
        if np.ndim(x) > 3:  # 降维
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)  # clip to [0, 1]
        x *= 255  # convert to RGB array
        # if K.image_dim_ordering() == 'th':
        # x = x.transpose((1, 2, 0)) # 改变图片通道排列
        x = np.clip(x, 0, 255).astype('float64')
        return x

    def target_category_loss(self, origin, predict):  # 计算分类的损失函数
        one_hot = torch.zeros(1, self.classNum).scatter_(1, predict, 1)
        return self.criterion(origin, one_hot)

    def keepInputDim(self, x):  # 由于格式需要
        return x

    def normalize_L2(self, x):  # L2正则化 : 平方，取均值，开方
        y = x.square()
        y = y.mean()
        y = y.sqrt()
        return x / y

    def conv_compute_gradients(self, loss, varList):  # 梯度计算函数
        loss.backward() # 使用随机梯度进行反传
        grads = self.model.(self.target_layer).bias.grad   # 求梯度 :: model继承自父类
        grad_list = []  # 然后再梯度中取非负值
        for var, grad in zip(varList, grads):
            grad_list.append(grad if grad is not None else torch.zeros_like(var))  # 如果是None则用0填充
        return grad_list

    def target_layer(self, x, predict):
        return self.target_category_loss(x, predict)  # 求处损失函数的相关值

    def get_datagrad(self, input):
        img_path = "cifar1.jpg"
        img_Raw = cv2.imread(img_path, 1)  # 读取灰色图像 :: 二维的
        # print(imgRaw)
        MCE = np.array([
            [0, 0, -1],
            [0, 0, -2]
        ], dtype=np.float64)
        img_Pre = (img_Raw)  # 训练前对图像进行处理
        img_ele = cv2.warpAffine(img_Pre, MCE, (512, 512))  # 插值处理，统一到512*512分辨率
        img_float64 = img_as_float(img_ele)  # 数值转化
        img_Tensor = torch.Tensor(img_float64)
        # img_norm = deprocess_image(img_float64) # 得到归一化的图像
        model = self.model(weights='imagenet')
        predictions = model.predict(img_Raw) # 需要确定一下模型的输入 :: 应该是model内部的一个函数
        predictions = torch.Tensor(predictions) # predictions 转成张量形式
        top_1 = decode_predictions(predictions)[0][0]
        print("Predictd class : ")
        print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
        predict = np.argmax(predictions)  # 得到样本的标签 softmax
        # 求处得到全连接层之前的解释信息
        layer = "block5_conv3"  # 这个是解析器需要知道的层
        modelOut = model.output
        x = self.target_layer(input,modelOut)
        model = Model(inputs=model.input, outputs=x) # 传入模型的输入和输出
        # model.summary()  # 打印模型的概述信息
        loss_CE = torch.nn.CrossEntropyLoss(model.output, target)

        # 得到神经网络最后一层的输出信息
        conv_output = [l for l in model.layers if l.name is layer][0].output
        # 计算损失函数对最后一层卷积的每一个输出的梯度
        
        grads = self.normalize_L2((self.conv_compute_gradients(loss, [conv_output])[0])  # 计算梯度并且正则化
        gradient_function = K.function(
            [model.input], [conv_output, grads])  # 得到一个计算梯度的函数

        output, grads_val = gradient_function([img_Tensor])  # 得到卷积层的输出和梯度
        output, grads_val = output[0, :], grads_val[0, :, :, :]  # 取第一行

        # 最重要的东西
        # [w1,w2,w3,...] 得到特征图关于分类标签的权重
        weights = np.mean(grads_val, axis=(0, 1))  # 得到卷积层到全连接层的权重权重

        # 按照output的形状建立一个临时矩阵
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)  # 相当于做一个ReLu激活
        heatmap = cam / np.max(cam)  # 产生热图

        # Return to BGR [0..255] from the preprocessed image
        img_float64 = img_float64[0, :]
        img_float64 -= np.min(img_float64)
        img_float64 = np.minimum(img_float64, 255)

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # 变成彩色图
        cam = np.float32(cam) + np.float32(img_float64)
        cam = 255 * cam / np.max(cam)  # 恢复像素
        cam = np.uint8(cam)
        cv2.imwrite("camGen.jpg", cam)
        cv2.imwrite("heatMap.jpg", heatmap)

        cv2.imshow("cam_jpg", cam)
        cv2.imshow("heatMap_jpg", heatmap)

        """
        plt.imshow(imgRaw,cmap = 'gray',interpolation = 'bicubic')
        plt.show()
        """
