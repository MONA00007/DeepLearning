#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def mean_squared_error(y, t):
    # 均方误差
    return np.sum((y-t)**2)/2


def cross_entropy_error(y, t):
    # 交叉熵误差
    delta = 1e-7  # 防止负无穷大
    return -np.sum(t*np.log(y+delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
