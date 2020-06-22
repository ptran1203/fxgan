
from collections import defaultdict, Counter
import keras.backend as K
import tensorflow as tf
import keras

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (
    UpSampling2D, Convolution2D,
    Conv2D, Conv2DTranspose
)
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.losses import mean_squared_error, cosine_similarity, KLDivergence
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Embedding, Dropout,
    BatchNormalization, Activation,
    Lambda,Layer, Add, Concatenate,
    Average,GaussianNoise,
    MaxPooling2D, AveragePooling2D,
    RepeatVector,GlobalAveragePooling2D,
)
from keras_contrib.losses import DSSIMObjective
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

from keras.applications.vgg16 import VGG16

from keras.utils import np_utils
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import os
import sys
import re
import numpy as np
import datetime
import pickle
import cv2
import utils

from google.colab.patches import cv2_imshow
from PIL import Image

K.common.set_image_dim_ordering('tf')

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))


class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)

        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = keras.layers.InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]

        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class FeatureNorm(keras.layers.Layer):
    def __init__(self, epsilon = 1e-4, norm = 'bn'):
        super(FeatureNorm, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def call(self, inputs):
        x, scale, bias = inputs
        # x = [batch, height, width, channels]
        N, H, W, C = x.shape

        # instance norm
        axis = [1, 2]
        if 'bn' in self.norm:
            print('Use Batch norm for FeatureNorm layer')
            axis = [0, 1, 2]

        mean = K.mean(x, axis = axis, keepdims = True)
        std = K.std(x, axis = axis, keepdims = True)
        norm = (x - mean) * (1 / (std + self.epsilon))

        broadcast_scale = K.reshape(scale, (-1, 1, 1, C))
        broadcast_bias = K.reshape(bias, (-1, 1, 1, C))

        return norm * broadcast_scale + broadcast_bias

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class BalancingGAN:
    D_RATE = 1
    attribute_loss_weight = 1
    def _triple_tensor(self, x):
        return Concatenate()([x,x,x])

    def _res_block(self,
                  x,
                  units = 64,
                  kernel_size = 3,
                  activation = 'leaky_relu',
                  norm = 'batch',
                  norm_var = [0,0]):
        scale, bias = norm_var
        if activation == 'leaky_relu':
            actv = LeakyReLU()
        else:
            actv = Activation(activation)

        def norm_layer(x):
            if norm == 'batch':
                x = BatchNormalization()(x)
            elif norm == 'in':
                x = InstanceNormalization()(x)
            else:
                x = FeatureNorm(norm=self.norm)([x, scale, bias])
            return x

        out = norm_layer(x)
        out = actv(out)
        out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)
        out = norm_layer(out)
        out = actv(out)

        out = Conv2D(K.int_shape(x)[-1], kernel_size, strides = 1, padding='same')(out)
        out = norm_layer(out)
        out = actv(out)
        out = Add()([out, x])
        return out

    def _upscale(self, x, interpolation='conv', units=64, kernel_size=5):
            if interpolation == 'conv':
                # use convolution
                x = Conv2DTranspose(units, kernel_size, strides=2, padding='same')(x)
                return x
            else:
                # use upsamling layer
                # nearest  or   bilinear
                x = Conv2D(units, kernel_size, strides=1, padding='same')(x)
                x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
                return x


    def build_attribute_encoder(self):
        """
        Mapping image to latent code
        """
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        kernel_size = 5

        x = Conv2D(32, kernel_size+2, strides = 1, padding='same')(image)
        x = self._norm()(x)
        x = Activation('relu')(x)
        # 32 * 32 * 32

        x = Conv2D(64, kernel_size, strides=1, padding='same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        # 16 * 16 * 64

        x = Conv2D(128, kernel_size, strides=2, padding='same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        # 8*8*128

        x = Conv2D(256, kernel_size, strides=2, padding='same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, kernel_size, strides=2, padding='same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)

        code = GlobalAveragePooling2D()(x)

        self.attribute_encoder = Model(
            inputs = image,
            outputs = code,
            name='attribute_encoder',
        )


    def attribute_net(self, images, channels):
        attr_features = []
        for i in range(self.k_shot):
            attr_features.append(self.latent_encoder(
                Lambda(lambda x: x[:, i,])(images)
            ))
        
        attr_feature = Average()(attr_features)

        scale = Dense(256, activation = 'relu')(attr_feature)
        scale = Dense(channels)(scale)

        bias = Dense(256, activation = 'relu')(attr_feature)
        bias = Dense(channels)(bias)

        return scale, bias

    def build_latent_encoder(self):
        fname = '/content/drive/My Drive/bagan/{}/latent_encoder_{}'.format(self.dataset,
                                                                            self.resolution)
        json_file = open(fname + '.json', 'r')
        model = json_file.read()
        json_file.close()
        self.latent_encoder = model_from_json(model)
        modified = os.path.getmtime(fname + '.json')
        print('Latent model modified at: ',
            datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S'))
        self.latent_encoder.load_weights(fname + '.h5')
        self.latent_encoder.trainable = False

    def latent_code(self, images, prediction=True):
        """
        Get prediction from latent encoder (Attribute code)
        """
        if prediction:
            return self.latent_encoder.predict(images)
        return self.latent_encoder(images)

    def latent_codes(self, k_shot_images, prediction=True):
        """
        Predict for k_shot images
        shape = (batch_size, K_shot, H, W, C)
        return: array with shape (batch_size, latent_code_size)
        """
        return np.array([
            np.mean(self.latent_code(i), axis=0) for i in k_shot_images
            ])


    def __init__(self, classes, loss_type = 'binary',
                adam_lr=0.00005, latent_size=100,
                res_dir = "./res-tmp", image_shape=[32, 32, 1],
                g_lr = 0.000005, norm = 'batch',
                resnet=False, beta_1 = 0.5,
                dataset = 'chest', attention=True,
                k_shot=5):
        self.classes = classes
        self.dataset = dataset
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[0]
        self.g_lr = g_lr
        self.resnet = resnet
        self.attention = attention
        self.k_shot = k_shot

        self.norm = norm
        self.loss_type = loss_type
        self._show_settings()

        if loss_type == 'binary':
            self.g_loss = keras.losses.BinaryCrossentropy()
            self.d_fake_loss = keras.losses.BinaryCrossentropy()
            self.d_real_loss = keras.losses.BinaryCrossentropy()
        elif loss_type == 'categorical':
            self.g_loss = 'sparse_categorical_crossentropy'
            self.d_fake_loss = 'sparse_categorical_crossentropy'
            self.d_real_loss = 'sparse_categorical_crossentropy'
        elif loss_type == 'hinge':
            self.g_loss = hinge_G_loss
            self.d_fake_loss = hinge_D_fake_loss
            self.d_real_loss = hinge_D_real_loss
        else:
            self.g_loss = wasserstein_loss
            self.d_fake_loss = wasserstein_loss
            self.d_real_loss = wasserstein_loss

        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = beta_1

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build networks
        self.build_perceptual_model()
        self.build_latent_encoder()
        self.build_attribute_encoder()
        self.build_discriminator()
        self.build_features_from_d_model()
        if self.resnet:
            print('INFO: Use resnet generator')
            self.build_resnet_generator()
        else:
            print('INFO: Use DCGAN generator')
            self.build_dc_generator()


        real_images = Input(shape=(self.resolution, self.resolution, self.channels))
        # Use VGG16 model -> channels = 3
        attr_images = Input(shape=(self.k_shot, self.resolution, self.resolution, 3))
        latent_code = Input(shape=(self.latent_size,))

        fake_images = Input(shape=(self.resolution, self.resolution, self.channels))

        real_output_for_d = self.discriminator([real_images, attr_images])
        fake_output_for_d = self.discriminator([fake_images, attr_images])

        self.discriminator_fake = Model(
            inputs = [fake_images, attr_images],
            outputs = fake_output_for_d,
            name='D_fake',
        )
        self.discriminator_fake.compile(
            optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics = ['accuracy'],
            loss = [self.d_fake_loss]
        )

        self.discriminator_real = Model(
            inputs = [real_images, attr_images],
            outputs = real_output_for_d,
            name='D_real',
        )
        self.discriminator_real.compile(
            optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics = ['accuracy'],
            loss = [self.d_real_loss]
        )

        # Define combined for training generator.
        real_images_for_G = Input((self.k_shot, self.resolution, self.resolution, 3))
        fake = self.generator([
            real_images_for_G, latent_code
        ])

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.features_from_d_model.trainable = False
        self.latent_encoder.trainable = False
        self.attribute_encoder.trainable = True

        aux_fake = self.discriminator([fake, real_images_for_G])

        negative_samples = Input((self.resolution,self.resolution,self.channels))
        fake_attribute = self.latent_encoder(self._triple_tensor(fake))

        ## real attr
        attr_features = []
        for i in range(self.k_shot):
            attr_features.append(self.latent_encoder(
                Lambda(lambda x: x[:, i,])(real_images_for_G)
            ))

        self.combined = Model(
            inputs=[real_images_for_G, negative_samples, latent_code],
            outputs=[aux_fake, fake_attribute],
            name = 'Combined',
        )

        # triplet function
        margin = 1.0
        d_pos = Average()([
            K.sum(K.square(fake_attribute - attr_feature), axis=1) \
                for attr_feature in attr_features
        ])
        d_neg = K.sum(K.square(
                fake_attribute -
                self.latent_encoder(self._triple_tensor(negative_samples))
                ), axis=1)

        triplet = K.maximum(d_pos - d_neg + margin, 0.0)


        # Feature matching from D net
        fm_features = [
            # Only use 1 channel
            self.features_from_d_model(Lambda(lambda x: x[:, i,:,:,:1])(real_images_for_G)) \
                for i in range(self.k_shot)
        ]

        fake_fm = self.features_from_d_model(fake)
        fm_D = Average()([
            K.square(fake_fm - fm_feature) \
                for fm_feature in fm_features
        ])

        self.combined.add_loss(self.attribute_loss_weight * triplet)
        self.combined.add_loss(K.mean(fm_D))

        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            metrics=['accuracy'],
            loss = [self.g_loss, 'mse'],
            loss_weights= [1.0, 0]
        )

    def build_resnet_generator(self):
        images = Input(shape=(self.k_shot, self.resolution, self.resolution, 3),
                        name = 'G_input')
        decoder_activation = Activation('relu')

        init_channels = 512
        latent_code = Input(shape=(128,), name = 'latent_code')

        latent = Dense(4 * 4 * init_channels)(latent_code)
        latent = self._norm()(latent)
        latent = decoder_activation(latent)
        latent = Reshape((4, 4, init_channels))(latent)

        kernel_size = 5
        norm_var = self.attribute_net(images, 512)

        de = self._res_block(latent, 512, kernel_size,
                            norm='fn',
                            norm_var=norm_var)
        de = self._res_block(de, 512, kernel_size,
                            norm='fn',
                            norm_var=norm_var)

        de = self._upscale(de, 'conv', 256, kernel_size)
        de = self._norm()(de)
        de = decoder_activation(de)

        if self.attention:
            de = SelfAttention(256)(de)

        de = self._upscale(de, 'conv', 128, kernel_size)
        de = self._norm()(de)
        de = decoder_activation(de)

        de = self._upscale(de, 'conv', 64, kernel_size)
        de = self._norm()(de)
        de = decoder_activation(de)

        final = Conv2DTranspose(self.channels, kernel_size, strides=2, padding='same')(de)
        outputs = Activation('tanh')(final)

        self.generator = Model(
            inputs = [images, latent_code],
            outputs = outputs,
            name='resnet_gen'
        )

    def build_dc_generator(self):
        def _transpose_block(x, units, activation, kernel_size=3, norm='batch', norm_var = [0,0]):
            scale, bias = norm_var
            def _norm_layer(x):
                if norm == 'batch':
                    x = BatchNormalization()(x)

                elif norm == 'in':
                    x = InstanceNormalization()(x)
                else:
                    x = FeatureNorm(norm=self.norm)([x, scale, bias])
                return x

            out = Conv2DTranspose(units, kernel_size, strides=2, padding='same')(x)
            out = _norm_layer(out)
            out = activation(out)
            # out = Dropout(0.3)(out)
            return out

        images = Input(shape=(self.k_shot, self.resolution, self.resolution, 3), name = 'G_input')
        decoder_activation = Activation('relu')
        kernel_size = 5
        init_channels = 512
        latent_code = Input(shape=(128,), name = 'latent_code')
        # attribute_code = self.attribute_code(image)

        # latent = Concatenate()([latent_code, attribute_code])
        latent = Dense(4 * 4 * init_channels)(latent_code)
        latent = self._norm()(latent)
        latent = decoder_activation(latent)
        latent = Reshape((4, 4, init_channels))(latent)

        de = _transpose_block(latent, 256, decoder_activation,
                             kernel_size, norm=self.norm,
                             norm_var=self.attribute_net(images, 256))
        if self.attention:
            de = SelfAttention(256)(de)

        de = _transpose_block(de, 128, decoder_activation,
                             kernel_size, norm=self.norm,
                             norm_var=self.attribute_net(images, 128))
        de = _transpose_block(de, 64, decoder_activation,
                             kernel_size, norm=self.norm,
                             norm_var=self.attribute_net(images, 64))

        final = Conv2DTranspose(self.channels, kernel_size, strides=2, padding='same')(de)
        outputs = Activation('tanh')(final)

        self.generator = Model(
            inputs = [images, latent_code],
            outputs = outputs,
            name='dc_gen'
        )


    def build_perceptual_model(self):
        """
        VGG16 model with imagenet weights
        """
        model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor = Input(shape=(self.resolution, self.resolution, 3)),
            input_shape = (self.resolution, self.resolution, 3)
        )
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        
        self.perceptual_model = model


    def plot_loss_his(self):
        def _get_arr(x, idx):
            return [i[idx] for i in x]

        def plot_g(train_g, test_g):
            plt.plot(_get_arr(train_g, 1), label='train mse')
            plt.plot(_get_arr(test_g, 1), label='test mse')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Generator')
            plt.legend()
            plt.show()

            plt.plot(_get_arr(train_g, 0), label='train adv')
            plt.plot(_get_arr(test_g, 0), label='test adv')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Generator')
            plt.legend()
            plt.show()

        def plot_d(train_d, test_d):
            plt.plot(train_d, label='train')
            plt.plot(test_d, label='test')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Discriminator')
            plt.legend()
            plt.show()

        train_d = self.train_history['disc_loss']
        train_g = self.train_history['gen_loss']
        test_d = self.test_history['disc_loss']
        test_g = self.test_history['gen_loss']

        if len(train_g) == 0:
            return 

        plot_g(train_g, test_g)
        plot_d(train_d, test_d)

    def _show_settings(self):
        print('\n=================== GAN Setting ==================\n')
        print('- Dataset: {}'.format(self.dataset))
        print('- Num of classes: {}'.format(self.nclasses))
        print('- Generator type: {}'.format('Resnet' if self.resnet else 'DCGAN'))
        print('- Self-Attention: {}'.format(self.attention))
        print('- K-shot: {}'.format(self.k_shot))
        print('- Adverasial loss: {}'.format(self.loss_type))
        if 'batch' in self.norm:
            norm_type = 'Batch norm'
        else:
            norm_type = 'Instance norm'
        print('- Normalization: {}'.format(norm_type))
        fn_norm = 'fn' in self.norm
        print('- Use feature normaliztion: {}'.format(fn_norm))
        print('\n==================================================\n')


    def _discriminator_feature(self, image, attr_image):
        resolution = self.resolution
        channels = self.channels

        kernel_size = 5
        x = Conv2D(64, kernel_size, strides=2, padding='same')(image)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, kernel_size, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
    
        if self.attention:
            x = SelfAttention(128)(x)

        x = Conv2D(256, kernel_size, strides=2, padding='same')(x)
        if 'D' in self.norm and 'fn' in self.norm:
            scale, bias = self.attribute_net(attr_image, 256)
            x = FeatureNorm(norm=self.norm)([x, scale, bias])
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(512, kernel_size, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        return Flatten()(x)


    def build_discriminator(self):
        resolution = self.resolution
        channels = self.channels

        image = Input(shape=(resolution, resolution, channels))
        attr_image = Input(shape=(self.k_shot, resolution, resolution, 3))

        features = self._discriminator_feature(image, attr_image)

        activation = 'sigmoid' if self.loss_type == 'binary' else 'linear'

        if self.loss_type == 'categorical':
            aux = Dense(self.nclasses + 1,
                        activation = 'softmax',
                        name='auxiliary')(features)
        else:
            aux = Dense(
                1, activation = activation,name='auxiliary'
            )(features)

        self.discriminator = Model(inputs=[image, attr_image],
                                   outputs=aux,
                                   name='discriminator')



    def generate_latent(self, c, size = 1):
        return np.array([
            np.random.normal(0, 1, self.latent_size)
            for i in c
        ])


    def build_features_from_d_model(self):
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        model_output = self.discriminator.layers[-2](image)
        self.features_from_d_model = Model(
            inputs = image,
            output = model_output,
            name = 'Feature_matching'
        )

    def _norm(self):
        return BatchNormalization() if 'batch' in self.norm else InstanceNormalization()

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []

        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################
            fake_size = crt_batch_size // self.nclasses
            f = self.generate_latent(range(crt_batch_size))
            k_shot_batch = bg_train.ramdom_kshot_images(self.k_shot, label_batch)
            for i in range(self.D_RATE):
                generated_images = self.generator.predict(
                    [
                        k_shot_batch,
                        f,
                    ],
                    verbose=0
                )

                # X, aux_y = self.shuffle_data(X, aux_y)
                fake_label = np.ones((crt_batch_size, 1))
                real_label = -np.ones((crt_batch_size, 1))
                real_label_for_d = -np.ones((crt_batch_size, 1))

                if self.loss_type == 'binary':
                    real_label *= 0
                    real_label_for_d *= 0
                if self.loss_type == 'categorical':
                    real_label = label_batch
                    real_label_for_d = label_batch
                    fake_label = np.full(crt_batch_size, self.nclasses)

                attr_images = bg_train.ramdom_kshot_images(self.k_shot, label_batch)
                loss_fake, acc_fake, *rest = \
                        self.discriminator_fake.train_on_batch([generated_images, attr_images],
                                                                fake_label)
                loss_real, acc_real, *rest = \
                        self.discriminator_real.train_on_batch([image_batch, attr_images],
                                                                real_label_for_d)
                loss = 0.5 * (loss_fake + loss_real)
                acc = 0.5 * (acc_fake + acc_real)

            epoch_disc_loss.append(loss)

            ################## Train Generator ##################
            f = self.generate_latent(range(crt_batch_size))
            negative_samples = bg_train.get_samples_by_labels(bg_train.other_labels(label_batch))
            real_attribute = self.latent_codes(k_shot_batch)
            [loss, d_loss, l_loss, *rest] = self.combined.train_on_batch(
                [k_shot_batch, negative_samples, f],
                [real_label, real_attribute],
            )

            epoch_gen_loss.append([d_loss, l_loss])

        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0),
        )

    def shuffle_data(self, data_x, data_y):
        rd_idx = np.arange(data_x.shape[0])
        np.random.shuffle(rd_idx)
        return data_x[rd_idx], data_y[rd_idx]

    def _get_lst_bck_name(self, element):
        # Find last bck name
        files = [
            f for f in os.listdir(self.res_dir)
            if re.match(r'bck_' + element, f)
        ]
        if len(files) > 0:
            fname = files[0]
            epoch = 0
            return epoch, fname

        else:
            return 0, None

    def init_gan(self):
        # Find last bck name
        epoch, generator_fname = self._get_lst_bck_name("generator")

        new_e, discriminator_fname = self._get_lst_bck_name("discriminator")
        if new_e != epoch:  # Reload error, restart from scratch
            return 0

        # Load last bck
        try:
            print('GAN weight initialized, train from epoch ', epoch)
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            return epoch

        # Return epoch
        except Exception as e:  # Reload error, restart from scratch (the first time we train we pass from here)
            print('Reload error, restart from scratch ', str(e))
            return 0

    def backup_point(self, epoch):
        # Bck
        if epoch == 0:
            return

        print('Save weights at epochs : ', epoch)
        generator_fname = "{}/bck_generator.h5".format(self.res_dir)
        discriminator_fname = "{}/bck_discriminator.h5".format(self.res_dir)

        self.generator.save(generator_fname)
        self.discriminator.save(discriminator_fname)

    def evaluate_d(self, test_x, test_y):
        y_pre = self.discriminator.predict(test_x)
        if y_pre[0].shape[0] > 1:
            y_pre = np.argmax(y_pre, axis=1)
        else:
            y_pre = utils.pred2bin(y_pre)
        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)  # shape=(12, 12)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def evaluate_g(self, test_x, test_y):
        y_pre = self.combined.predict(test_x)
        if y_pre[0].shape[0] > 1:
            y_pre = np.argmax(y_pre, axis=1)
        else:
            y_pre = pred2bin(y_pre)

        cm = metrics.confusion_matrix(y_true=test_y[0], y_pred=y_pre)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            self.autoenc_epochs = 100

            # Initialization
            print("init gan")
            start_e = self.init_gan()
            # self.init_autoenc(bg_train)
            print("gan initialized, start_e: ", start_e)

            crt_c = 0
            # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                        np.full(10, crt_c))
            f = self.generate_latent(range(10))
    
            img_samples = np.array([
                [
                    act_img_samples[:, 0,:,:,:1],
                    self.generator.predict([
                        act_img_samples,
                        f,
                    ]),
                ]
            ])
            for crt_c in range(1, min(self.nclasses, 3)): # more 3 classes
                # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                            np.full(10, crt_c))
                new_samples = np.array([
                    [
                        act_img_samples[:, 0,:,:,:1],
                        self.generator.predict([
                            act_img_samples,
                            f,
                        ]),
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            utils.show_samples(img_samples)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss = self._train_one_epoch(bg_train)

                # Get Test samples
                test_size = 100
                random_ids = np.arange(bg_test.dataset_y.shape[0])
                np.random.shuffle(random_ids)
                random_ids = random_ids[:test_size]
                test_batch_x = bg_test.dataset_x[random_ids]
                test_batch_y = bg_test.dataset_y[random_ids]
                k_shot_test_batch = bg_test.ramdom_kshot_images(self.k_shot, test_batch_y)
                f = self.generate_latent(range(test_size))

                generated_images = self.generator.predict(
                    [
                        k_shot_test_batch,
                        f
                    ],
                    verbose=False
                )

                X = np.concatenate([test_batch_x, generated_images])
    
                aux_y = np.concatenate([
                    np.full(test_size, 0),
                    np.full(test_size, 1)
                ])

                fake_label = np.ones((test_size, 1))
                real_label = -np.ones((test_size, 1))

                if self.loss_type == 'binary':
                    real_label *= 0
                if self.loss_type == 'categorical':
                    real_label = rand_y
                    fake_label = np.full(test_size, self.nclasses)

                X = [test_batch_x, generated_images]
                Y = [fake_label, real_label]

                loss_fake, acc_fake, *rest = \
                        self.discriminator_fake.evaluate([generated_images, k_shot_test_batch],
                                                        fake_label, verbose=False)
                loss_real, acc_real, *rest = \
                        self.discriminator_real.evaluate([test_batch_x, k_shot_test_batch],
                                                        real_label, verbose=False)
                test_disc_loss = 0.5 * (loss_fake + loss_real)
                test_disc_acc = 0.5 * (acc_fake + acc_real)

                negative_samples = bg_train.get_samples_by_labels(bg_train.other_labels(test_batch_y))
                real_attribute = self.latent_codes(k_shot_test_batch)
                [_, gen_d_loss, gen_latent_loss, *_] = self.combined.evaluate(
                    [
                        k_shot_test_batch,
                        negative_samples,
                        f
                    ],
                    [real_label, real_attribute],
                    verbose = 0
                )

                if e % 25 == 0:
                    # self.evaluate_d(np.concatenate([X[0], X[1]], axis=0), np.concatenate(Y, axis=0))
                    # self.evaluate_g(
                    #     [
                    #         bg_test.dataset_x,
                    #         negative_samples,
                    #         f,
                            
                    #     ],
                    #     [real_label, real_attribute],
                    # )

                    crt_c = 0
                    # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c))

                    f = self.generate_latent(range(10))
                    img_samples = np.array([
                        [
                            act_img_samples[:, 0,:,:,:1],
                            self.generator.predict([
                                act_img_samples,
                                f,
                                
                            ]),
                        ]
                    ])
                    for crt_c in range(1, min(self.nclasses, 3)):
                        # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c))
                        f = self.generate_latent(range(10))
                        new_samples = np.array([
                            [
                                act_img_samples[:, 0,:,:,:1],
                                self.generator.predict([
                                    act_img_samples,
                                    f,
                                    
                                ]),
                            ]
                        ])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)

                    utils.show_samples(img_samples)

                    # calculate attribute distance
                    self.plot_loss_his()
                    self.plot_feature_distr(bg_train)

                if e % 100 == 0:
                    self.backup_point(e)

                self.interval_process(e)


                print("- D_loss {}, G_adv_loss {} G_mse_loss {} - {}".format(
                    train_disc_loss, train_gen_loss[0], train_gen_loss[1],
                    datetime.datetime.now() - start_time
                ))

                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append([gen_d_loss, gen_latent_loss])

            self.trained = True

    def plot_feature_distr(self, bg, size=500):
        x, y = bg.dataset_x, bg.dataset_y
        real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, 0))
        fakes = self.generator.predict([real,
                                        self.generate_latent(range(size))])
        fake_labels = [np.full((size,), 'fake of 0')]

        for classid in range(1, min(self.nclasses, 5)):
            real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, classid))
            fake = self.generator.predict([real, self.generate_latent(range(size))])
            fakes = np.concatenate([fakes, fake])
            fake_labels.append(np.full((size,), 'fake of {}'.format(classid)))

        # latent_encoder
        imgs = np.concatenate([x, fakes])
        labels = np.concatenate([
            np.full((x.shape[0],), 'real'),
            np.full((fakes.shape[0],), 'fake'),
        ])

    
        utils.plot_data_space(imgs, labels, self.features_from_d_model, 'fake real space')
        labels = np.concatenate([y, np.concatenate(fake_labels)])
        utils.plot_data_space(imgs, labels, self.latent_encoder, 'latent encoder')

    def interval_process(self, epoch, interval = 20):
        if epoch % interval != 0:
            return
        # do bussiness thing