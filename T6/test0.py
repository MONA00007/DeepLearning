from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from test0_0 import max_features, input_train, y_train

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
'''
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
'''
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
