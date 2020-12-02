#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pylab as plt


def func(x):
    return x[0]**2+x[1]**2


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for ide in range(x.size):
        tmp_val = x[ide]
        x[ide] = tmp_val+h
        fxh1 = f(x)
        x[ide] = tmp_val-h
        fxh2 = f(x)
        grad[ide] = (fxh1-fxh2)/(2*h)
        x[ide] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        x -= lr*numerical_gradient(f, x)

    return x


init_x = np.array([-3.0, 4.0])
t = gradient_descent(func, init_x, lr=0.1)
print(t)
