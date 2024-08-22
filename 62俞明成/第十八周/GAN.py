from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# TensorFlow过滤掉警告，‌只输出错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 潜在空间的维度
        self.latent_dim = 100
        #  learning_rate,beta1
        optimizer = Adam(0.0002, 0.5)

        # 构建判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 构建生成器
        self.generator = self.build_generator()

        # 将噪声作为输入并生成图像
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 设置判别器不进行训练
        self.discriminator.trainable = False

        # 判别器将生成的图像作为输入并确定有效性
        validity = self.discriminator(img)

        # 组合模型
        # 训练生成器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        # 输入，输出
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 加载mnist数据集
        (x_train, _), (_, _) = mnist.load_data()

        # 缩放到[-1,1]
        x_train = x_train / 127.5 - 1.
        # 在做后一个维度上增加一个维度channel
        x_train = np.expand_dims(x_train, axis=3)

        # 真
        valid = np.ones((batch_size, 1))
        # 生成
        fake = np.zeros((batch_size, 1))
        best_loss = float('inf')
        for epoch in range(epochs):
            # ---------------------
            #  训练判别器
            # ---------------------
            # 从训练集随机选取batch_size个样本
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            images = x_train[idx]
            # 生成一个潜在向量的批次，用于输入到生成器中生成新的图像
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 使用生成器生成一批新的图像
            gen_images = self.generator.predict(noise)

            # d_loss_real：用真实图像 images 和标签 valid（全为1）来训练判别器。
            # d_loss_fake：用生成的图像 gen_images 和标签 fake（全为0）来训练判别器。
            # d_loss：判别器的总损失是 d_loss_real 和 d_loss_fake 的均值。
            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  训练生成器
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 在不更新判别器的情况下，更新生成器的权重，从而使生成器能够生成更逼真的图像，以欺骗判别器
            # 因为要使生成器的损失最小，生成器需要欺骗判别器，因此判别器标签为1，即valid。
            g_loss = self.combined.train_on_batch(noise, valid)

            # 损失值d_loss[0], 准确度100 * d_loss[1]
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            # 每隔 sample_interval 轮次，保存当前生成器生成的图像样本
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        self.generator.save('generator.h5')

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_images = self.generator.predict(noise)

        # [-1,1]->[0,1]
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
