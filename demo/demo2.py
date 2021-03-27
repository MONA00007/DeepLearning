import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
# 预处理图像
model = load_model('cats_and_dogs_small_2.h5')
# model.summary()


img_path = 'D:\\BaiduNetdiskDownload\\train\\train\\cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
'''
plt.imshow(img_tensor[0])
plt.show()
'''