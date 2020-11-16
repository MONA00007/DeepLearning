#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pylab as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def step_func(x):
    y = x > 0
    return y.astype(np.int)


x = np.array([[1, 2], [3, 4]])
y = np.array([[3, 5], [2, 9]])
print(x.shape)
print(np.dot(x, y))
