from keras.models import Sequential
from keras import layers
from test1 import float_data
model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32),
                               input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mae')
#history=model.fit_generator()