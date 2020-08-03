
from collections import defaultdict, Counter
import keras.backend as K
import tensorflow as tf
import keras

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (
    UpSampling2D, Convolution2D,
    Conv2D, Conv2DTranspose
)
import math
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.losses import mean_squared_error
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
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import re
import numpy as np
import datetime
import pickle
import cv2
import utils
import logger
from const import BASE_DIR

K.common.set_image_dim_ordering('tf')

from PIL import Image

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))

def safe_average(list_inputs):
    if len(list_inputs) == 1:
        return list_inputs[0]
    return Average()(list_inputs)

def l2_distance(a, b):
    return np.mean(np.square(a - b))

def cosine_sim(a, b):
    return - (np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def cosine_sim_op(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def actv(activation):
    if activation == 'leaky_relu':
        return LeakyReLU()
    return Activation(activation)

def norm_layer(norm, x):
    if norm is None:
        return x
    if norm == 'batch':
        x = BatchNormalization()(x)
    elif norm == 'in':
        x = InstanceNormalization()(x)
    return x

class DAGAN:
    D_RATE = 1
    def _triple_tensor(self, x):
        if x.shape[-1] == 3:
            return x
        return Concatenate()([x,x,x])

    def _up_resblock(self,
                  x,
                  units = 64,
                  kernel_size = 3,
                  activation = 'leaky_relu',
                  norm = 'batch',
                  attr_image=None):

        interpolation = 'nearest'

        out = norm_layer(norm, x)
        out = actv(activation)(out)
        out = UpSampling2D(size=(2, 2), interpolation=interpolation)(out)
        out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)

        out = norm_layer(norm, out)
        out = actv(activation)(out)
        out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)

        x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
        x = Conv2D(units, 1, strides = 1, padding='same')(x)

        return Add()([out, x])

    def _donw_resblock(self,
                      x,
                      units=64,
                      kernel_size=3,
                      activation='leaky_relu',
                      norm=None):

        out = actv(activation)(x)
        out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)

        out = actv(activation)(out)
        out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)
        out = AveragePooling2D(pool_size=(2, 2))(out)

        x = Conv2D(units, 1, strides = 1, padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        return Add()([out, x])

    def _conv_block(self, x, units, 
                kernel_size=3,activation='relu',
                norm='batch', transpose=True, strides=2):
        if transpose:
            x = Conv2DTranspose(units, kernel_size, strides=strides, padding='same')(x)
        else:
            x = Conv2D(units, kernel_size, strides=strides, padding='same')(x)
        x = norm_layer(norm, x)
        x = actv(activation)(x)
        return x


    def show_samples_for_class(self,bg,classid, mode='00'):
        """
        Show K-samples + 10 - k generated image based on K
        mode params: including 2 digit numbers (string) xy
                     x = 0 => difference image
                     y = 0 => difference latent
                     and 1 mean the same image/latent
        """
        mode_x, mode_y = int(mode[0]), int(mode[1])

        samples = 10 - self.k_shot
        if mode_x == 0:
            print("difference images")
            support_images = bg.ramdom_kshot_images(self.k_shot,
                                                    [classid] * samples,False)
        else:
            print("same images")
            support_images = bg.ramdom_kshot_images(self.k_shot,
                                                    [classid],False)
            support_images = np.repeat(support_images,
                                        samples,
                                        axis=0)
        if mode_y == 0:
            print("difference latent")
            latent = self.generate_latent([classid] * samples)
        else:
            print("same latent")
            latent = np.repeat(self.generate_latent([classid]), samples, axis=0)
    
        generated_images = self.generate(support_images, latent)
        utils.show_samples(support_images)
        utils.show_samples(generated_images)


    def gen_for_class(self, bg, bg_test=None, classid=0,size=1000):
        total = None
        for i in range(1000):
            labels = [classid] * size
            labels = np.array(labels)
            latent = self.generate_latent(labels)
            if classid in bg.classes:
                support = bg.ramdom_kshot_images(self.k_shot,
                                                np.full(size, classid),False)
            else:
                if bg_test is None:
                    raise("bg_test is None, please give it, boi")

                support = bg_test.ramdom_kshot_images(self.k_shot,
                                                np.full(size, classid),False)

            gen = self.generate(support, latent)
            if total is None:
                total = gen
            else:
                total = np.concatenate([total, gen], axis=0)
            
            print("total ", len(total))
            if len(total) >= size:
                total = total[:size]
                break

        print("done class {}, size {}\n".format(classid, len(total)))
        return total, np.array([classid] * len(total))

    def gen_augment_data(self, bg, bg_test=None, size=1000):
        total = None
        labels = None
        counter = dict(Counter(
            np.concatenate([bg.dataset_y, bg_test.dataset_y]) \
                    if bg_test is not None else bg.dataset_y
        ))
        max_ = max(counter.values())
        for i in bg.classes:
            acctual_size = max((max_ - counter[i]), 0)
            if acctual_size == 0:
                print("Skip class", i)
                continue
            gen, label = self.gen_for_class(bg, bg_test, i, acctual_size)
            if total is None:
                total = gen
                labels = label
            else:
                total = np.concatenate([total, gen], axis=0)
                labels = np.concatenate([labels, label], axis=0)
        if bg_test is not None:
            for i in bg_test.classes:
                acctual_size = max((max_ - counter[i]), 0)
                gen, label = self.gen_for_class(bg, bg_test, i, acctual_size)
                if total is None:
                    total = gen
                    labels = label
                else:
                    total = np.concatenate([total, gen], axis=0)
                    labels = np.concatenate([labels, label], axis=0)

        print("Done all ", len(total))
        return total, labels


    def __init__(self, classes, loss_type = 'binary',
                adam_lr=0.00005, latent_size=100,
                res_dir = "./res-tmp", image_shape=[32, 32, 1],
                g_lr = 0.000005, norm = 'batch', beta_1 = 0.5,
                dataset = 'chest',
                k_shot=5,
                env="colab",
                upsample="dc",
                ):
        self.classes = classes
        self.dataset = dataset
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[0]
        self.g_lr = g_lr
        self.k_shot = k_shot
        self.env=env
        self.upsample=upsample
        # normal: sampling from normal distribution
        # code: sampling from latent code distribution (computed by classifier)
        self.norm = norm
        self.loss_type = loss_type

        if loss_type == 'binary':
            self.g_loss = keras.losses.BinaryCrossentropy()
            self.d_fake_loss = keras.losses.BinaryCrossentropy()
            self.d_real_loss = keras.losses.BinaryCrossentropy()
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
        self.build_discriminator()
        self.build_generator()

        latent_code = Input(shape=(self.latent_size,))
        real_images = Input(shape=(self.resolution, self.resolution, self.channels))
        similiar_images = Input(shape=(self.resolution, self.resolution, self.channels))
        fake_images = Input(shape=(self.resolution, self.resolution, self.channels))

        real_output_for_d = self.discriminator([real_images, similiar_images])
        fake_output_for_d = self.discriminator([fake_images, similiar_images])

        self.discriminator_fake = Model(
            inputs = [fake_images, similiar_images],
            outputs = fake_output_for_d,
            name='D_fake',
        )
        self.discriminator_fake.compile(
            optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss = self.d_fake_loss
        )

        self.discriminator_real = Model(
            inputs = [real_images, similiar_images],
            outputs = real_output_for_d,
            name='D_real',
        )
        self.discriminator_real.compile(
            optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss = self.d_real_loss
        )

        # Define combined for training generator.
        real_images_for_G = Input((self.resolution, self.resolution, self.channels))
        fake = self.generator([
            real_images_for_G, latent_code
        ])

        self.discriminator.trainable = False
        self.generator.trainable = True

        aux_fake = self.discriminator([fake, real_images_for_G])
        self.combined = Model(
            inputs=[real_images_for_G, latent_code],
            outputs=aux_fake,
            name = 'Combined',
        )

        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            loss = self.g_loss,
        )

    def encoder(self, kernel_size ,img):
        def conv_block(units, kernel_size, ip):
            x = Conv2D(units, kernel_size, strides=2, padding='same')(ip)
            x = LeakyReLU()(x)
            x = norm_layer(self.norm, x)
            return x

        connections = [img]
        channels = 64
        width = img.shape[-2]
        i = 0
        while width != 4:
            connections.append(conv_block(channels, kernel_size, connections[i]))
            width //= 2
            channels *= 2
            i += 1

        return connections[1:]


    def build_generator(self):
        init_channels = self.resolution * 4
        latent_code = Input(shape=(self.latent_size,), name = 'latent_code')
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        activation = 'relu'
        kernel_size = 3

        encoded = self.encoder(kernel_size, image)

        latent = Dense(4 * 4 * init_channels)(latent_code)
        latent = Reshape((4, 4, init_channels))(latent)

        latent1 = Dense(8 * 8 * init_channels//2)(latent_code)
        latent1 = Reshape((8, 8, init_channels//2))(latent1)

        latent2 = Dense(16 * 16 * init_channels//4)(latent_code)
        latent2 = Reshape((16, 16, init_channels//4))(latent2)
        latents = [latent1,latent2]

        de = Concatenate()([encoded[-1], latent])

        # how much loops to upscale to the resolution?
        for i in range(int(math.log(self.resolution / 4) // math.log(2)) - 1):
            init_channels //= 2
            if self.upsample == "dc":
                de = self._conv_block(de, init_channels, kernel_size,
                                    activation=activation,
                                    norm=self.norm)
            else:
                de = self._conv_block(de, init_channels, kernel_size,
                                    activation=activation,
                                    norm=self.norm,
                                    transpose=False, strides=1)
                de = self._upsample(de)
            de = Add()([encoded[-(i + 2)], de])
            if i < 2:
                de = Concatenate()([de, latents[i]])

        if self.upsample == 'dc':
            final = self._conv_block(de, self.channels, kernel_size,
                            activation='tanh',
                            norm=None)
        else:
            final = self._conv_block(de, init_channels, kernel_size,
                                    activation=activation,
                                    norm=self.norm,
                                    transpose=False, strides=1)
            final = self._upsample(final)
            final = self._conv_block(final, self.channels, kernel_size,
                                    activation='tanh',
                                    norm=None,
                                    transpose=False, strides=1)

        self.generator = Model(
            inputs = [image, latent_code],
            outputs = final,
            name='dc_gen'
        )

    def _upsample(self, x, interpolation='nearest'):
        return UpSampling2D(size=(2, 2), interpolation=interpolation)(x)


    def plot_loss_his(self):
        def plot_g(train_g):
            plt.plot(train_g, label='train adv')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Generator')
            plt.legend()
            plt.show()

        def plot_d(train_d):
            plt.plot(train_d, label='train')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Discriminator')
            plt.legend()
            plt.show()

        train_d = self.train_history['disc_loss']
        train_g = self.train_history['gen_loss']

        if len(train_g) == 0:
            return 

        plot_g(train_g)
        plot_d(train_d)


    def _discriminator_feature(self, image):
        resolution = self.resolution
        channels = self.channels

        kernel_size = 3

        x = Conv2D(64, kernel_size, strides=2, padding='same')(image)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, kernel_size, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        channels = 256
        # downsample to 4
        while x.shape[-2] != 4:
            x = Conv2D(channels, kernel_size, strides=2, padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)
            channels *= 2

        x = Flatten()(x)
        return x


    def build_discriminator(self):
        resolution = self.resolution
        channels = self.channels

        image = Input(shape=(resolution, resolution, channels))
        same_image = Input(shape=(resolution, resolution, channels))

        combined_img = Concatenate()([image, same_image])

        features = self._discriminator_feature(image)
        features = Dropout(0.4)(features)
        aux = Dense(1, name='auxiliary')(features)
        self.discriminator = Model(inputs=[image, same_image],
                                   outputs=aux,
                                   name='discriminator')


    def generate_latent(self, c, size = 1):
        return np.array([
            np.random.normal(0, 1, self.latent_size)
            for i in c
        ])


    def _norm(self):
        return BatchNormalization() if 'batch' in self.norm else InstanceNormalization()

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################
            fake_size = crt_batch_size // self.nclasses
            f = self.generate_latent(label_batch)
            k_shot_batch = bg_train.ramdom_kshot_images(self.k_shot, label_batch,False)
            generated_images = self.generate(image_batch, f)

            fake_label = np.ones((crt_batch_size, 1))
            real_label = -np.ones((crt_batch_size, 1))

            loss_fake = \
                    self.discriminator_fake.train_on_batch([generated_images, image_batch],
                                                            fake_label)
            loss_real = \
                    self.discriminator_real.train_on_batch([k_shot_batch, image_batch],
                                                            real_label)
            loss = 0.5 * (loss_fake + loss_real)

            epoch_disc_loss.append(loss)

            ################## Train Generator ##################
            f = self.generate_latent(label_batch)
            gloss = self.combined.train_on_batch(
                [
                    image_batch,
                    f
                ],
                real_label,
            )

            epoch_gen_loss.append(gloss)

        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0),
        )


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

        # Load last bck
        load_dir = self.res_dir if self.env == "colab" else "/content"
        try:
            self.generator.load_weights(os.path.join(load_dir, generator_fname))
            logger.info("generator weigths loaded")
            self.discriminator.load_weights(os.path.join(load_dir, discriminator_fname))
            logger.info("discriminator weigths loaded")
            return epoch

        except Exception as e:
            e = str(e)
            logger.warn('Reload error, restart from scratch, ' + e)
            return 0


    def backup_point(self, epoch):
        # Bck
        if epoch == 0:
            return

        print('Save weights at epochs : ', epoch)
        save_dir = self.res_dir if self.env == "colab" else "/content"
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
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues,figsize=(8,8))
        plt.show()


    def evaluate_g(self, test_x, test_y):
        fakes = self.generate(test_x[0], test_x[1])
        y_pre = self.discriminator.predict(fakes)
        if y_pre[0].shape[0] > 1:
            y_pre = np.argmax(y_pre, axis=1)
        else:
            y_pre = pred2bin(y_pre)

        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues,figsize=(8,8))
        plt.show()


    def generate(self, image, latent):
        return self.generator.predict([
            image, latent
        ])

    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            # Initialization
            print("init gan")
            start_e = self.init_gan()
            # self.init_autoenc(bg_train)
            print("gan initialized, start_e: ", start_e)

            crt_c = 0
            # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                        np.full(10, crt_c), False)
            f = self.generate_latent([crt_c] * 10)
            img_samples = np.array([
                [
                    act_img_samples,
                    self.generate(act_img_samples, f)
                ]
            ])
            for crt_c in range(1, min(self.nclasses, 3)): # more 3 classes
                # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                            np.full(10, crt_c), False)
                new_samples = np.array([
                    [
                        act_img_samples,
                        self.generate(act_img_samples, f)
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            utils.show_samples(img_samples)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('DAGAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss = self._train_one_epoch(bg_train)

                if e % (50 // (self.resolution // 32)) == 0:
                    self.backup_point(e)
                    crt_c = 0
                    # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c), False)

                    f = self.generate_latent([crt_c] * 10)
                    img_samples = np.array([
                        [
                            act_img_samples,
                            self.generate(act_img_samples, f)
                        ]
                    ])
                    for crt_c in range(1, min(self.nclasses, 3)):
                        # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c),False)
                        f = self.generate_latent([crt_c] * 10)
                        new_samples = np.array([
                            [
                                act_img_samples,
                                self.generate(act_img_samples, f)
                            ]
                        ])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)

                    utils.show_samples(img_samples)

                    self.plot_loss_his()

                if e % (100 // (self.resolution // 32)) == 0:
                    self.backup_point(e)


                print("- D_loss {}, G_loss {} - {}".format(
                    train_disc_loss, train_gen_loss,
                    datetime.datetime.now() - start_time
                ))

                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)

            self.trained = True

    def plot_feature_distr(self, bg, size=50):
        x, y = bg.dataset_x, bg.dataset_y
        real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, bg.classes[0]),False)

        fakes = self.generate(real, self.generate_latent([0] * size))

        fake_labels = [np.full((size,), 'fake of 0')]

        for classid in bg.classes[1:5]:
            real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, classid))
            fake = self.generate(real, self.generate_latent([classid] * size),False)
            fakes = np.concatenate([fakes, fake])
            fake_labels.append(np.full((size,), 'fake of {}'.format(classid)))

        # latent_encoder
        imgs = np.concatenate([x, fakes])
        labels = np.concatenate([
            np.full((x.shape[0],), 'real'),
            np.full((fakes.shape[0],), 'fake'),
        ])

        labels = np.concatenate([y, np.concatenate(fake_labels)])
        utils.scatter_plot(imgs, labels, self.latent_encoder, 'latent encoder')
