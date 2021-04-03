from keras.applications import inception_v3
from keras import backend

backend.set_learning_phase(0)

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
model.summary()
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}
