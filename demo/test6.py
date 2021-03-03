from keras import layers, models

model = models.Sequential()
model.add(layers.Convolution2D(
    32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

model.summary()
