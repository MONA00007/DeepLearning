#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def step_func(x):
    y = x > 0
    return y.astype(np.int)


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_func(x)
plt.plot(x, y1)
plt.plot(x, y2, linestyle='--')
plt.ylim(-0.1, 1.1)
plt.show()
