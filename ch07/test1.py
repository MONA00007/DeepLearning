#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import common.util
import numpy as np
import sys
import os
sys.path.append(os.pardir)
print(sys.path)

x1 = np.random.rand(1, 3, 7, 7)
col1 = util.im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)
