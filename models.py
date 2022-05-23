from time import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.initializers import RandomUniform
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class DownsizeOnly(Model):
    """
    This is not a real model, it only downsizes the training sample to where we start with, used for generating pictures for the report
    """
    def __init__(self, latent_dim):
        super(DownsizeOnly, self).__init__()
        self.latent_dim = latent_dim
        self.out = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")
        ])

    def call(self, x):
        out = self.out(x)
        return out

class NearestReference(Model):
    """
    This is not a NN but a reference model using Nearest Neighbour upsampling. Our NN is trying to beat this.
    """
    def __init__(self, latent_dim):
        super(NearestReference, self).__init__()
        self.latent_dim = latent_dim
        self.pipeline = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest"),
            layers.UpSampling2D(size = (4, 4), interpolation = "nearest")
        ])

    def call(self, x):
        return self.pipeline(x)

class BilinearReference(Model):
    """
    This is not a NN but a reference model using Bilinear upsampling. Our NN is trying to beat this.
    """
    def __init__(self, latent_dim):
        super(BilinearReference, self).__init__()
        self.latent_dim = latent_dim
        self.pipeline = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest"),
            layers.UpSampling2D(size = (4, 4), interpolation = "bilinear")
        ])

    def call(self, x):
        return self.pipeline(x)

class BicubicReference(Model):
    """
    This is not a NN but a reference model using Bicubic upsampling. Our NN is trying to beat this.
    """
    def __init__(self, latent_dim):
        super(BicubicReference, self).__init__()
        self.latent_dim = latent_dim
        self.pipeline = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest"),
            layers.experimental.preprocessing.Resizing(height = latent_dim[1], width = latent_dim[2], interpolation = "bicubic")
        ])

    def call(self, x):
        return self.pipeline(x)

class GaussianReference(Model):
    """
    This is not a NN but a reference model using Gaussian upsampling. Our NN is trying to beat this.
    """
    def __init__(self, latent_dim):
        super(GaussianReference, self).__init__()
        self.latent_dim = latent_dim
        self.pipeline = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest"),
            layers.experimental.preprocessing.Resizing(height = latent_dim[1], width = latent_dim[2], interpolation = "gaussian")
        ])

    def call(self, x):
        return self.pipeline(x)

class AutoencoderA(Model):
    def __init__(self, latent_dim):
        super(AutoencoderA, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest"),
            layers.UpSampling2D(size = (4, 4), interpolation = "nearest"),
            layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu", input_shape = latent_dim[1:4]),
            layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPooling2D((2, 2), strides = 2, padding = "valid"),
            layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPooling2D((2, 2), strides = 2, padding = "valid"),
            layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.UpSampling2D(size = (2, 2), interpolation = "nearest"),
            layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.UpSampling2D(size = (2, 2), interpolation = "nearest"),
            layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 3, kernel_size = (3, 3), padding = "same", activation = "relu")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def AutoencoderB(latent_dim):
    # encoder
    input_ = layers.Input(shape = latent_dim[1:4])
    layer1 = layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")(input_)
    layer2 = layers.UpSampling2D(size = (4, 4), interpolation = "nearest")(layer1)
    layer3 = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu", input_shape = latent_dim[1:4])(layer2)
    layer4 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(layer3)
    layer5 = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(layer4)
    layer6 = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(layer5)
    layer7 = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(layer6)
    layer8 = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(layer7)
    # decoder
    layer9 = layers.UpSampling2D(size = (2, 2), interpolation = "nearest")(layer8)
    layer10 = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(layer9)
    layer11 = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(layer10)
    layer12 = layers.Add()([layer6, layer11])
    layer13 = layers.UpSampling2D(size = (2, 2), interpolation = "nearest")(layer12)
    layer14 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(layer13)
    layer15 = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(layer14)
    layer16 = layers.Add()([layer3, layer15])
    layer17 = layers.Conv2D(filters = 3, kernel_size = (3, 3), padding = "same", activation = "relu")(layer16)
    return Model(input_, layer17, name = "AutoencoderB")



def AutoencoderD(latent_dim):
    """
    This model beats all reference implementations
    """
    # encoder
    input_ = layers.Input(shape = latent_dim[1:4])
    l = layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")(input_)
    l = layers.UpSampling2D(size = (4, 4), interpolation = "bilinear")(l)
    l_a = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu", input_shape = latent_dim[1:4])(l)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l_a)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l)
    l_b = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l_b)
    l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l)
    l_c = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 256, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    # decoder
    l = layers.Conv2D(filters = 256, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_c])
    l = layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_b])
    l = layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_a])
    out_ = layers.Conv2D(filters = 3, kernel_size = (1, 1), padding = "same", activation = "relu")(l)
    return Model(input_, out_, name = "AutoencoderD")

def AutoencoderE(latent_dim):
    """
    This model beats all reference implementations
    """
    # encoder
    input_ = layers.Input(shape = latent_dim[1:4])
    l = layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")(input_)
    l_0 = layers.UpSampling2D(size = (4, 4), interpolation = "bilinear")(l)
    l_a = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu", input_shape = latent_dim[1:4])(l_0)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l_a)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 64, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l)
    l_b = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l_b)
    l = layers.Conv2D(filters = 128, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l)
    l_c = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 256, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    # decoder
    l = layers.Conv2D(filters = 256, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_c])
    l = layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
    l = layers.Conv2D(filters = 128, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_b])
    l = layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
    l = layers.Conv2D(filters = 64, kernel_size = (5, 5), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
    l = layers.Add()([l, l_a])
    out_ = layers.Conv2D(filters = 3, kernel_size = (1, 1), padding = "same", activation = "relu")(l)
    return Model(input_, out_, name = "AutoencoderE")

def psnr(y_real, y_predicted, max_val = 1):
    return tf.image.psnr(y_real, y_predicted, max_val = 1)

def ResidualCNN(latent_dim):
    """
    This model beats all reference implementations
    """
    # encoder
    input_ = layers.Input(shape = latent_dim[1:4])
    l = layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")(input_)
    l_a = layers.Conv2D(filters = 64, kernel_size = (1, 1), padding = "same", activation = "relu", input_shape = latent_dim[1:4])(l)
    before_residuals = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(l_a)
    residual_blocks = [before_residuals]
    prev = before_residuals
    for _ in range(16):
        r1 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(prev)
        a1 = layers.Add()([prev, r1])
        r2 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a1)
        a2 = layers.Add()([a1, r2])
        r3 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a2)
        a3 = layers.Add()([a2, r3])
        r4 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a3)
        a4 = layers.Add()([a3, r4])
        r5 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a4)
        a5 = layers.Add()([a4, r5])
        r6 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a5)
        a6 = layers.Add()([a5, r6])
        r7 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a6)
        a7 = layers.Add()([a6, r7])
        r8 = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(a7)
        a8 = layers.Add()([a7, r8])
        c1 = layers.Concatenate()([prev, a1, a2, a3, a4, a5, a6, a7, a8])
        l = layers.Conv2D(filters = 64, kernel_size = (1, 1), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(c1)
        o1 = layers.Add()([prev, l])
        residual_blocks.append(o1)
        prev = o1
    c = layers.Concatenate()(residual_blocks)
    l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu", kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = 1.5), bias_initializer = "zeros")(c)
    a = layers.Add()([l, before_residuals])
    l = layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3), padding = "same", strides = (4, 4))(a)
    out_ = layers.Conv2D(filters = 3, kernel_size = (1, 1), padding = "same", activation = "relu")(l)
    return Model(input_, out_, name = "ResidualCNN")

class GAN():
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)

    @staticmethod
    def GeneratorFactory(latent_dim):
        input_ = layers.Input(shape = latent_dim[1:4])
        l = layers.experimental.preprocessing.Resizing(height = latent_dim[1] // 4, width = latent_dim[2] // 4, interpolation = "nearest")(input_)
        l = layers.experimental.preprocessing.Resizing(height = latent_dim[1], width = latent_dim[2], interpolation = "bicubic")(l)
        l_a = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu", input_shape = latent_dim[1:4])(l)
        l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l_a)
        l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l)
        l_b = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.MaxPooling2D((2, 2), strides = 2, padding = "valid")(l_b)
        l_c = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        # decoder
        l = layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Add()([l, l_c])
        l = layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
        l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Add()([l, l_b])
        l = layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = "same", strides = (2, 2))(l)
        l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(l)
        l = layers.Add()([l, l_a])
        out_ = layers.Conv2D(filters = 3, kernel_size = (1, 1), padding = "same", activation = "relu")(l)
        return Model(input_, out_, name = "GAN.generator")

    @staticmethod
    def DiscriminatorFactory(latent_dim):
        initial_weights = tf.keras.initializers.RandomNormal(stddev = 0.02)
        initial_gamma = tf.keras.initializers.RandomNormal(mean = 1., stddev = 0.02)
        leakyRelu = layers.LeakyReLU(alpha = 0.02)
        input_ = layers.Input(shape = latent_dim[1:4])
        l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(input_)
        l = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(l)
        l = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(l)
        back = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(back)
        l = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(l)
        l = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = leakyRelu, kernel_initializer = initial_weights)(l)
        l = layers.BatchNormalization(axis = 1, gamma_initializer = initial_gamma)(l)
        l = layers.Add()([back, l])
        l = layers.Flatten()(l)
        l = layers.Dropout(0.2)(l)
        out_ = layers.Dense(1, kernel_initializer = initial_weights, activation = "sigmoid")(l)
        return Model(input_, out_, name = "GAN.discriminator")

    @staticmethod
    def discriminator_loss(y_real, y_predicted):
        real_loss = GAN.cross_entropy(tf.ones_like(y_real), y_real)
        fake_loss = GAN.cross_entropy(tf.zeros_like(y_predicted), y_predicted)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(y_real, y_predicted):
        return GAN.cross_entropy(tf.ones_like(y_predicted), y_predicted) + GAN.mse(y_predicted, y_real)

    def __init__(self, latent_dim, batch_size = 4):
        self.name = "GAN"
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()
        self.generator = GAN.GeneratorFactory(self.latent_dim)
        self.discriminator = GAN.DiscriminatorFactory(self.latent_dim)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer, discriminator_optimizer = self.discriminator_optimizer, generator = self.generator, discriminator = self.discriminator)

    def compile(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        pass

    def load_weights(self, *args, **kwargs):
        self.checkpoint.restore(tf.train.latest_checkpoint("checkpoints"))

    def predict(self, data_iterator, *args, **kwargs):
        X_predicted = self.generator(next(validation_iterator), train = False)
        for _ in range(len(validation_iterator)):
            batch = next(validation_iterator)[0]
            X_predicted = tf.concat([X_predicted, self.generator(batch, train = False)], axis = 0)
        return X_predicted

    def summary(self):
        pass

    @tf.function
    def train_step(self, X):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_images = self.generator(X, training = True)
            real_output = self.discriminator(X, training = True)
            fake_output = self.discriminator(generated_images, training = True)
            generator_loss = GAN.generator_loss(X, generated_images)
            discriminator_loss = GAN.discriminator_loss(real_output, fake_output)
        gradients_of_generator = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
        gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def fit(self, train_iterator, validation_data, epochs, shuffle, callbacks):
        validation_iterator = validation_data
        for epoch in range(1):
            validation_loss = 0
            start = time()
            for _ in range(len(train_iterator)):
                with tf.GradientTape() as generator_tape:
                    batch = next(train_iterator)
                    generated_images = self.generator(batch[0], training = True)
                    generator_loss = GAN.generator_loss(batch[1], generated_images)
                gradients_of_generator = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            for index in range(len(validation_iterator)):
                batch = next(validation_iterator)
                validation_predicted = self.generator(batch[0], training = False)
                mse_validation_loss = GAN.mse(validation_predicted, batch[1])
                validation_loss = ((validation_loss * index) + mse_validation_loss) / (index + 1)
            print("Validation MSE loss: {}".format(validation_loss))
            print("Time for epoch {} is {} sec".format(epoch + 1, time() - start))
        for epoch in range(epochs):
            validation_loss = 0
            start = time()
            for _ in range(len(train_iterator)):
                self.train_step(next(train_iterator)[0])
            for index in range(len(validation_iterator)):
                batch = next(validation_iterator)
                validation_predicted = self.generator(batch[0], training = False)
                mse_validation_loss = GAN.mse(validation_predicted, batch[1])
                validation_loss = ((validation_loss * index) + mse_validation_loss) / (index + 1)
            self.checkpoint.save(file_prefix = "checkpoints/GAN_")
            print("Validation MSE loss: {}".format(validation_loss))
            print("Time for epoch {} is {} sec".format(epoch + 1, time() - start))


    def evaluate(self, test_iterator):
        test_loss = 0
        for index in range(len(test_iterator)):
            batch = next(test_iterator)
            test_predicted = self.generator(batch[0], training = False)
            mse = tf.keras.losses.MSE(reduction = tf.keras.losses.Reduction.SUM)
            mse_validation_loss = mse(test_predicted, batch[1])
            test_loss = ((test_loss * index) + mse_validation_loss) / (index + 1)
        print("Test loss: %.5f" % (test_loss))

