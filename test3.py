#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt


# 阶跃函数的实现
def step_func(x):
    y = x > 0
    return y.astype(np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_func(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
