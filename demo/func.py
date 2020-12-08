#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.datasets import reuters
from keras import layers, models
import numpy as np


def indexToWord(train_data):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key)
                               for key, value in word_index.items()])
    decoded_newswire = ''.join(
        [reverse_word_index.get(i-3, '?')for i in train_data[0]])
    return decoded_newswire


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.

    return results


def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
