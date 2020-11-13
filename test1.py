#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('timg.jpg')
plt.imshow(img)
plt.show()
