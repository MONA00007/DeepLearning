#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.datasets import boston_housing
# from keras import layers, models
from func import build_model
import matplotlib.pyplot as plt
import numpy as np


(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
'''
# k折验证
k = 4
num_val_samples = len(train_data)//k  # 先除，再向下取整
num_epochs = 500
# all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]  # 划区域
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]], axis=0)
    model = build_model(train_data)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # print(history.history.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    # all_scores.append(val_mae)

# print(all_scores)
# print(np.mean(all_scores))
average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                       for i in range(num_epochs)]


def smooth_curve(points, factor=0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+point*(1-factor))
        else:
            smooth_points.append(point)
    return smooth_points


plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Mae')
plt.show()
plt.clf()
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Mae')
plt.show()
'''
model = build_model(train_data)
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
