#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common.functions import softmax, cross_entropy_error
import numpy as np
import sys
import os
sys.path.append(os.pardir)
# from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
