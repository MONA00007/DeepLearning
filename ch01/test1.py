#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

sys.path.append(os.pardir)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                  normalize=False)

img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img)
img_show(img)
