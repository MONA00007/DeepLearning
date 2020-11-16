#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataset.mnist import load_mnist
import sys
import os

sys.path.append(os.pardir)

(X_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                  normalize=False)
