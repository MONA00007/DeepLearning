#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import layers, models
import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical
from func import vectorize_sequences
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)
# 将训练和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 将训练和测试标签向量化
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
