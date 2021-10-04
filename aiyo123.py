from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, BatchNormalization, Input, Add, ZeroPadding2D, MaxPooling2D, Flatten, Dense
# 按 He et al. (2016) 定义两类残差块。


def identity_block(X, f, channels):
    '''
    卷积块--(等值函数)--卷积块
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # 主通路
    # 块1
    X = Conv2D(filters=F1, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 块 2
    X = Conv2D(filters=F2, kernel_size=(f, f),
               strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 块 3
    X = Conv2D(filters=F3, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    # 跳跃连接
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, channels, s=2):
    '''
    卷积块--(卷积块)--卷积块
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # 主通路
    # 块1
    X = Conv2D(filters=F1, kernel_size=(1, 1),
               strides=(s, s), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 块2
    X = Conv2D(filters=F2, kernel_size=(f, f),
               strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 块3
    X = Conv2D(filters=F3, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    # 跳跃连接
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1),
                        strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# ===== ResNet ===== #


class Paper():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256

        self.mask_height = 64
        self.mask_width = 64

        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # 生成器判别器
        self.generator = self.build_generator()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 准备联合训练
        self.discriminator.trainable = False
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)
        valid = self.discriminator(gen_missing)
        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def build_generator(self):
        masked_img = Input(shape=(256, 256, 3))  # 输入256x256的RGB图像
        # 0填充
        X = ZeroPadding2D((3, 3))(masked_img)
        # 主通路
        X = Conv2D(64, (7, 7), strides=(2, 2))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2))(X)  # 64,64,64
        # 残差块1
        X = convolutional_block(X, 3, [64, 64, 256], s=1)
        X = identity_block(X, 3, [64, 64, 256])  # 64,64,256
        # 残差块2
        X = convolutional_block(X, 3, [128, 128, 512])
        X = identity_block(X, 3, [128, 128, 512])  # 32,32,512
        # 残差块3
        X = convolutional_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        # 残差块4
        X = convolutional_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        # 反卷积
        # 块5
        X = UpSampling2D()(X)
        X = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)
        # 块6
        X = UpSampling2D()(X)
        X = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)
        # 块7
        X = UpSampling2D()(X)
        X = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)  # 64,64,256
        # 块8
        X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)
        # 块9
        X = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = Activation('tanh')(X)
        gen_missing = BatchNormalization(momentum=0.8)(X)
        # 全局均值池化
        # X = keras.layers.AveragePooling2D(pool_size=(1, 1), padding='same')(X)
        # X = keras.layers.Flatten()(X)
        # # 输出分类（按ImageNet为1000个分类）
        # OUT = keras.layers.Dense(1000, activation='softmax')(X)
        # Create model
        ResNet = Model(masked_img, gen_missing)  # masked_img gen_missing
        ResNet.summary()
        return Model(masked_img, gen_missing)  # masked_img gen_missing

    def build_discriminator(self):  # GAN 判别器
        # 输入图像位64,64,3
        img = Input(shape=(64, 64, 3))
        X = Conv2D(256, kernel_size=(3, 3),
                   strides=(1, 1), padding='same')(img)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)

        X = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)

        X = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)

        X = Conv2D(2048, kernel_size=(3, 3), strides=(2, 2), padding='same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(momentum=0.8)(X)
        # model=Model(img,X)
        # model.summary()
        X = Flatten()(X)
        validity = Dense(1, activation='softmax')(X)
        # 完成了输入64*64*3 图片 输出真是概率
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # if os.path.exists('saved_model/discriminator_weights.hdf5') and os.path.exists(
        #         'saved_model/generator_weights.hdf5'):
        #     self.discriminator.load_weights('saved_model/discriminator_weights.hdf5')
        #     self.generator.load_weights('saved_model/generator_weights.hdf5')
        #     print('-------------load the model-----------------')

        X_train = []

        list = glob.glob(r'train_images/arch/*.jpg')
        for ll in list:
            im = cv2.imread(ll)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            X_train.append(im)
        X_train = np.array(X_train)

        print('X_train.shape', X_train.shape,
              "———————————————————数据集加载完成——————————")

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 训练判别器
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = imgs / 175.5 - 1.  # -1 - 1
            # 随机抽取batchsize个真实图像

            masked_imgs, missing_parts, _ = self.mask_randomly(imgs)
            # masked_imgs就代表了遮挡的batch个图像
            # missing_parts就代表了丢失的batch个图像块

            gen_missing = self.generator.predict(masked_imgs)
            # 通过真假两个图训练判别器
            d_loss_real = self.discriminator.train_on_batch(
                missing_parts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 返回损失值和准确率

            # 训练生成器
            g_loss = self.combined.train_on_batch(
                masked_imgs, [missing_parts, valid])
            # 在这里 使用的是 MSE ADloss

            # 打印损失值以及准确率
            print("%d [D loss: %f, acc: %.2f%%] [G loss mse: %f, ad loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
            # d_loss[0]判别器损失, d_loss[1]准确率, g_loss[0]联合模型的重建损失, g_loss[1]联合模型的对抗损失
            if epoch % sample_interval == 0:
                # 随机生成5个整数
                idx = np.random.randint(0, X_train.shape[0], 5)
                imgs = X_train[idx]
                imgs = imgs / 127.5 - 1.
                self.sample_images(epoch, imgs)
            if epoch % 1000 == 0:
                self.save_model()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

    def sample_images(self, epoch, imgs):
        r, c = 3, 5
        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.generator.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        # imshow 绘制原图 遮挡图 和修复图
        for i in range(c):
            axs[0, i].imshow(imgs[i, :, :])
            axs[0, i].axis('off')
            axs[1, i].imshow(masked_imgs[i, :, :])
            axs[1, i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2, i].imshow(filled_in)
            axs[2, i].axis('off')
        fig.savefig("image/%d.png" % epoch, dpi=256)
        plt.close()

    def mask_randomly(self, imgs):
        y1 = np.random.randint(
            0, self.img_rows-self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(
            0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        # 复制原图 待遮挡（这里只需要完成像素置0即可完成遮挡）
        masked_imgs = np.empty_like(imgs)

        # 丢失区域内容大小尺寸定义完毕 （这里只需要将丢失的像素点复制进来）
        missing_parts = np.empty(
            (imgs.shape[0], self.mask_height, self.mask_width, self.channels))

        for i, img in enumerate(imgs):
            masked_img = img.copy()  # 首先复制原图 也就是准备完成遮挡单个图
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]  # 随机生成的每个遮挡坐标
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0  # 置0操作 完成遮挡
            masked_imgs[i] = masked_img  # 存入 masked_imgs
        return masked_imgs, missing_parts, (y1, y2, x1, x2)


# 编译模型
# opt = keras.optimizers.Adam(lr=0.001, decay=0.0)
# ResNet.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
# # 输出模型结构
# keras.utils.plot_model(ResNet, show_shapes=True, show_layer_names=False)
# #
# ResNet.fit_generator(...)  # 训练模型
if __name__ == '__main__':
    paper = Paper()
    paper.train(epochs=30000, batch_size=32, sample_interval=50)
