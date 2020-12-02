#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from common.gradient import numerical_gradient
from common.functions import cross_entropy_error, softmax
from dataset.mnist import load_mnist
import numpy as np
import sys
import os
sys.path.append(os.pardir)


class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp-(x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out

        return dx


class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.x = x
        out = np.dot(x, self.W)+self.b

        return out

    def backward(self, dout):

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:

    def __init__(self):
        self.t = None
        self.loss = None
        self.y = None

    def forward(self, x, t):

        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):

        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size

        return dx


class TwoLayerNet:

    def __init__(self, input_size, hidden_size,
                 output_size, weight_init_std=0.01):
        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()  # 有序字典
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # 进行识别（推理)
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # 计算损失函数
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # 计算识别精度
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t)/float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        # 通过数值微分计算梯度,速度较慢，一般用于验算反向法
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # 通过反向传播法计算梯度，速度很快，用于实际应用

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50,
                      output_size=10)

iters_num = 10000  # 总量
train_size = x_train.shape[0]
batch_size = 100  # 小批量包
learning_rate = 0.1  # 学习率
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)  # 高速版！

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grads[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc,test acc | "+str(train_acc)+","+str(test_acc))
