#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import load_mnist
import sys
import os


sys.path.append(os.pardir)

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True,
                                                  normalize=True)

print(x_train.shape)
print(t_train.shape)
