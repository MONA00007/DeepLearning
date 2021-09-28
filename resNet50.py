import tensorflow as tf
from tensorflow import keras


# 按 He et al. (2016) 定义两类残差块。
def identity_block(X, f, channels):
    '''
    卷积块--(等值函数)--卷积块
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # 主通路
    # 块1
    X = keras.layers.Conv2D(filters=F1, kernel_size=(
        1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 块 2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(
        f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 块 3
    X = keras.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    # 跳跃连接
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X


def convolutional_block(X, f, channels, s=2):
    '''
    卷积块--(卷积块)--卷积块
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # 主通路
    # 块1
    X = keras.layers.Conv2D(filters=F1, kernel_size=(
        1, 1), strides=(s, s), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 块2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(
        f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 块3
    X = keras.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    # 跳跃连接
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1),
                                     strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X


# ===== ResNet-50 ===== #
IN_GRID = keras.layers.Input(shape=(256, 256, 3))  # 输入256x256的RGB图像
# 0填充
X = keras.layers.ZeroPadding2D((3, 3))(IN_GRID)
# 主通路
X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(X)
X = keras.layers.BatchNormalization(axis=3)(X)
X = keras.layers.Activation('relu')(X)
X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
# 残差块1
X = convolutional_block(X, 3, [64, 64, 256])
X = identity_block(X, 3, [64, 64, 256])
X = identity_block(X, 3, [64, 64, 256])
# 残差块2
X = convolutional_block(X, 3, [128, 128, 512])
X = identity_block(X, 3, [128, 128, 512])
X = identity_block(X, 3, [128, 128, 512])
X = identity_block(X, 3, [128, 128, 512])
# 残差块3
X = convolutional_block(X, 3, [256, 256, 1024], s=2)
X = identity_block(X, 3, [256, 256, 1024])
X = identity_block(X, 3, [256, 256, 1024])
X = identity_block(X, 3, [256, 256, 1024])
X = identity_block(X, 3, [256, 256, 1024])
X = identity_block(X, 3, [256, 256, 1024])
# 残差块4
X = convolutional_block(X, 3, [512, 512, 2048])
X = identity_block(X, 3, [512, 512, 2048])
X = identity_block(X, 3, [512, 512, 2048])
# 全局均值池化
X = keras.layers.AveragePooling2D(pool_size=(1, 1), padding='same')(X)
X = keras.layers.Flatten()(X)
# 输出分类（按ImageNet为1000个分类）
OUT = keras.layers.Dense(1000, activation='softmax')(X)
# Create model
ResNet50 = keras.models.Model(inputs=IN_GRID, outputs=OUT)
ResNet50.summary()
print(type(ResNet50))
print("---------------------")
print(OUT)
# 编译模型
opt = keras.optimizers.Adam(lr=0.001, decay=0.0)
ResNet50.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=opt, metrics=['accuracy'])
# # 输出模型结构
# keras.utils.plot_model(ResNet50, show_shapes=True, show_layer_names=False)
# #
# ResNet50.fit_generator(...)  # 训练模型
