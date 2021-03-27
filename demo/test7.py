from keras import layers, models

model_no_max_pool = models.Sequential()
model_no_max_pool.add(layers.Convolution2D(
    32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_no_max_pool.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model_no_max_pool.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model_no_max_pool.summary()
