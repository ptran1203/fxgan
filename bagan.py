

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

from google.colab.patches import cv2_imshow
from PIL import Image

K.common.set_image_dim_ordering('tf')

class BalancingGAN:
    def plot_loss_his(self):
        train_d = self.train_history['disc_loss']
        train_g = self.train_history['gen_loss']
        test_d = self.test_history['disc_loss']
        test_g = self.test_history['gen_loss']
        plt.plot(train_d, label='train_d_loss')
        plt.plot(train_g, label='train_g_loss')
        plt.plot(test_d, label='test_d_loss')
        plt.plot(test_g, label='test_g_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def plot_acc_his(self):
        train_d = self.train_history['disc_acc']
        train_g = self.train_history['gen_acc']
        test_d = self.test_history['disc_acc']
        test_g = self.test_history['gen_acc']
        plt.plot(train_d, label='train_d_acc')
        plt.plot(train_g, label='train_g_acc')
        plt.plot(test_d, label='test_d_acc')
        plt.plot(test_g, label='test_g_acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def build_generator(self, latent_size, init_resolution=8):
        resolution = self.resolution
        channels = self.channels
        init_channels = 256
        cnn = Sequential()

        cnn.add(Dense(init_channels * init_resolution * init_resolution, input_dim=latent_size))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU())
        cnn.add(Reshape((init_resolution, init_resolution, init_channels)))
       
        crt_res = init_resolution
        # upsample
        while crt_res < resolution/2:
            cnn.add(Conv2DTranspose(
                init_channels, kernel_size = 5, strides = 2, padding='same'))
            cnn.add(LeakyReLU(alpha=0.02))
            init_channels //= 2
            crt_res = crt_res * 2
            assert crt_res <= resolution,\
                "Error: final resolution [{}] must equal i*2^n. Initial resolution i is [{}]. n must be a natural number.".format(resolution, init_resolution)
        cnn.add(Conv2DTranspose(
                    1, kernel_size = 5,
                    strides = 2, padding='same',
                    activation='tanh'))

        latent = Input(shape=(latent_size, ))

        fake_image_from_latent = cnn(latent)
        self.generator = Model(inputs=latent, outputs=fake_image_from_latent)

    def _build_common_encoder(self, image, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(128, (5, 5), padding='same', strides=(2, 2)))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))
            
        cnn.add(Conv2D(256, (5, 5), padding='same', strides=(2, 2)))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(512, (5, 5), padding='same', strides=(2, 2)))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        return cnn(image)

    # latent_size is the innermost latent vector size; min_latent_res is latent resolution (before the dense layer).
    def build_reconstructor(self, latent_size, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution, channels))
        features = self._build_common_encoder(image, min_latent_res)
        # Reconstructor specific
        latent = Dense(latent_size, activation='linear')(features)
        self.reconstructor = Model(inputs=image, outputs=latent)

    def build_discriminator(self, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution, channels))
        features = self._build_common_encoder(image, min_latent_res)
        # Discriminator specific
        features = Dropout(0.4)(features)
        aux = Dense(self.nclasses+1,
                    activation='softmax',
                    name='auxiliary')(features)
        self.discriminator = Model(inputs=image, outputs=aux)

    def generate_from_latent(self, latent):
        return self.generator(latent)


    def generate(self, c, bg=None):  # c is a vector of classes
        latent = self.generate_latent(c, bg)
        return self.generator.predict(latent)

    def evaluate_g(self, test_x, test_y):
        y_pre = self.combined.predict(test_x)
        y_pre = np.argmax(y_pre, axis=1)
        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)  # shape=(12, 12)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def generate_latent(self, c, bg=None, n_mix=10):  # c is a vector of classes
        return np.array([
            np.random.multivariate_normal(self.means[e], self.covariances[e])
            for e in c
        ])


    def gen_for_class(self, bg, classid,size=1000):
        total = None
        for i in range(1000):
            labels = [classid] * size
            labels = np.array(labels)
            latent = self.generate_latent(labels)
            print("Predict...")
            gen = self.generator.predict(latent)
            d_outputs = self.discriminator.predict(gen)
            d_outputs = np.argmax(d_outputs, axis=1)
            print(Counter(d_outputs))
            to_keep = np.where(labels == d_outputs)[0]
            gen = gen[to_keep]
            if total is None:
                total = gen
            else:
                total = np.concatenate([total, gen], axis=0)
            
            if len(total) >= size:
                break

        print("done class {}, size {}".format(classid, len(total)))
        return total, np.array([classid] * len(total))

    def gen_augment_data(self, bg, size=1000):
        total = None
        labels = None
        for i in bg.classes:
            gen , label = self.gen_for_class(bg, i, size)
            if total is None:
                total = gen
                labels = label
            else:
                total = np.concatenate([total, gen], axis=0)
                labels = np.concatenate([labels, label], axis=0)
        
        print("Done all ", len(total))
        return total, labels

    def discriminate(self, image):
        return self.discriminator(image)

    def __init__(self, classes, target_class_id,
                 # Set dratio_mode, and gratio_mode to 'rebalance' to bias the sampling toward the minority class
                 # No relevant difference noted
                 dratio_mode="uniform", gratio_mode="uniform",
                 adam_lr=0.00005, latent_size=100,
                 res_dir = "./res-tmp", image_shape=[3,32,32], min_latent_res=8,
                 autoenc_epochs=100):
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.classes = classes
        self.target_class_id = target_class_id  # target_class_id is used only during saving, not to overwrite other class results.
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[1]
        self.autoenc_epochs =autoenc_epochs

        self.min_latent_res = min_latent_res

        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build generator
        self.build_generator(latent_size, init_resolution=min_latent_res)

        latent_gen = Input(shape=(latent_size, ))

        # Build discriminator
        self.build_discriminator(min_latent_res=min_latent_res)
        self.discriminator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics=['accuracy'],
            loss='sparse_categorical_crossentropy'
        )

        # Build reconstructor
        self.build_reconstructor(latent_size, min_latent_res=min_latent_res)
        self.reconstructor.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

        # Define combined for training generator.
        fake = self.generator(latent_gen)

        self.discriminator.trainable = False
        self.reconstructor.trainable = False
        self.generator.trainable = True
        aux = self.discriminate(fake)

        self.combined = Model(inputs=latent_gen, outputs=aux)

        self.combined.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics=['accuracy'],
            loss='sparse_categorical_crossentropy'
        )

        # Define initializer for autoencoder
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.reconstructor.trainable = True

        img_for_reconstructor = Input(shape=(self.resolution, self.resolution,self.channels))
        img_reconstruct = self.generator(self.reconstructor(img_for_reconstructor))
        self.autoenc_0 = Model(inputs=img_for_reconstructor, outputs=img_reconstruct)
        self.autoenc_0.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

    def _biased_sample_labels(self, samples, target_distribution="uniform"):
        distribution = self.class_uratio
        if target_distribution == "d":
            distribution = self.class_dratio
        elif target_distribution == "g":
            distribution = self.class_gratio
            
        sampled_labels = np.full(samples,0)
        sampled_labels_p = np.random.normal(0, 1, samples)
        for c in list(range(self.nclasses)):
            mask = np.logical_and((sampled_labels_p > 0), (sampled_labels_p <= distribution[c]))
            sampled_labels[mask] = self.classes[c]
            sampled_labels_p = sampled_labels_p - distribution[c]

        return sampled_labels

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        epoch_disc_acc = []
        epoch_gen_acc = []

        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]
            ################## Train Discriminator ##################
            fake_size = int(np.ceil(crt_batch_size * 1.0/self.nclasses))
    
            # sample some labels from p_c, then latent and images
            sampled_labels = self._biased_sample_labels(fake_size, "d")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            generated_images = self.generator.predict(latent_gen, verbose=0)

            X = np.concatenate((image_batch, generated_images))
            aux_y = np.concatenate((label_batch, np.full(len(sampled_labels) , self.nclasses )), axis=0)

            loss, acc = self.discriminator.train_on_batch(X, aux_y)
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################
            sampled_labels = self._biased_sample_labels(fake_size + crt_batch_size, "g")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            loss, acc = self.combined.train_on_batch(latent_gen, sampled_labels)
            epoch_gen_loss.append(loss)
            epoch_gen_acc.append(acc)

        # return statistics: generator loss,
        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0),
            np.mean(np.array(epoch_disc_acc), axis=0),
            np.mean(np.array(epoch_gen_acc), axis=0),
        )

    def _set_class_ratios(self):
        self.class_dratio = np.full(self.nclasses, 0.0)
        # Set uniform
        target = 1/self.nclasses
        self.class_uratio = np.full(self.nclasses, target)
        
        # Set gratio
        self.class_gratio = np.full(self.nclasses, 0.0)
        for c in range(self.nclasses):
            if self.gratio_mode == "uniform":
                self.class_gratio[c] = target
            elif self.gratio_mode == "rebalance":
                self.class_gratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown gmode " + self.gratio_mode)
                exit()
                
        # Set dratio
        self.class_dratio = np.full(self.nclasses, 0.0)
        for c in range(self.nclasses):
            if self.dratio_mode == "uniform":
                self.class_dratio[c] = target
            elif self.dratio_mode == "rebalance":
                self.class_dratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown dmode " + self.dratio_mode)
                exit()

        # if very unbalanced, the gratio might be negative for some classes.
        # In this case, we adjust..
        if self.gratio_mode == "rebalance":
            self.class_gratio[self.class_gratio < 0] = 0
            self.class_gratio = self.class_gratio / sum(self.class_gratio)
            
        # if very unbalanced, the dratio might be negative for some classes.
        # In this case, we adjust..
        if self.dratio_mode == "rebalance":
            self.class_dratio[self.class_dratio < 0] = 0
            self.class_dratio = self.class_dratio / sum(self.class_dratio)

    def init_autoenc(self, bg_train, gen_fname=None, rec_fname=None):
        if gen_fname is None:
            generator_fname = "{}/{}_decoder.h5".format(self.res_dir, self.target_class_id)
        else:
            generator_fname = gen_fname
        if rec_fname is None:
            reconstructor_fname = "{}/{}_encoder.h5".format(self.res_dir, self.target_class_id)
        else:
            reconstructor_fname = rec_fname

        multivariate_prelearnt = False

        # Preload the autoencoders
        if os.path.exists(generator_fname) and os.path.exists(reconstructor_fname):
            print("BAGAN: loading autoencoder: ", generator_fname, reconstructor_fname)
            self.generator.load_weights(generator_fname)
            self.reconstructor.load_weights(reconstructor_fname)

            # load the learned distribution
            if os.path.exists("{}/{}_means.npy".format(self.res_dir, self.target_class_id)) \
                    and os.path.exists("{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)):
                multivariate_prelearnt = True

                cfname = "{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)
                mfname = "{}/{}_means.npy".format(self.res_dir, self.target_class_id)
                print("BAGAN: loading multivariate: ", cfname, mfname)
                self.covariances = np.load(cfname)
                self.means = np.load(mfname)

        else:
            print("BAGAN: training autoencoder")
            autoenc_train_loss = []
            for e in range(self.autoenc_epochs):
                print('Autoencoder train epoch: {}/{}'.format(e+1, self.autoenc_epochs))
                autoenc_train_loss_crt = []
                for image_batch, label_batch in bg_train.next_batch():

                    autoenc_train_loss_crt.append(self.autoenc_0.train_on_batch(image_batch, image_batch))
                autoenc_train_loss.append(np.mean(np.array(autoenc_train_loss_crt), axis=0))

            autoenc_loss_fname = "{}/{}_autoencoder.csv".format(self.res_dir, self.target_class_id)
            with open(autoenc_loss_fname, 'w') as csvfile:
                for item in autoenc_train_loss:
                    csvfile.write("%s\n" % item)

            self.generator.save(generator_fname)
            self.reconstructor.save(reconstructor_fname)

        layers_r = self.reconstructor.layers
        layers_d = self.discriminator.layers

        for l in range(1, len(layers_r)-1):
            layers_d[l].set_weights( layers_r[l].get_weights() )

        # Organize multivariate distribution
        if not multivariate_prelearnt:
            print("BAGAN: computing multivariate")
            self.covariances = []
            self.means = []

            for c in range(self.nclasses):
                imgs = bg_train.dataset_x[bg_train.per_class_ids[c]]
                latent = self.reconstructor.predict(imgs)

                self.covariances.append(np.cov(np.transpose(latent)))
                self.means.append(np.mean(latent, axis=0))

            self.covariances = np.array(self.covariances)
            self.means = np.array(self.means)

            # save the learned distribution
            cfname = "{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)
            mfname = "{}/{}_means.npy".format(self.res_dir, self.target_class_id)
            print("BAGAN: saving multivariate: ", cfname, mfname)
            np.save(cfname, self.covariances)
            np.save(mfname, self.means)
            print("BAGAN: saved multivariate")


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
        try:
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            print('GAN weight initialized, train from epoch ', epoch)
            return epoch

        except Exception as e:
            e = str(e)
            logger.warn('Reload error, restart from scratch ' + e)
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


    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            # Class actual ratio
            self.class_aratio = bg_train.get_class_probability()

            # Class balancing ratio
            self._set_class_ratios()

            # Initialization
            print("BAGAN init_autoenc")
            self.init_autoenc(bg_train)
            print("BAGAN autoenc initialized, init gan")
            start_e = self.init_gan()
            print("BAGAN gan initialized, start_e: ", start_e)

            crt_c = 0
            act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            img_samples = np.array([
                [
                    act_img_samples,
                    self.generator.predict(
                        self.reconstructor.predict(
                            act_img_samples
                        )
                    ),
                    self.generate_samples(crt_c, 10, bg_train)
                ]
            ])
            for crt_c in range(1, 3):
                act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                new_samples = np.array([
                    [
                        act_img_samples,
                        self.generator.predict(
                            self.reconstructor.predict(
                                act_img_samples
                            )
                        ),
                        self.generate_samples(crt_c, 10, bg_train)
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            shape = img_samples.shape
            img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))

            utils.show_samples(img_samples)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc = self._train_one_epoch(bg_train)

                if False:
                    # Test: # generate a new batch of noise
                    nb_test = bg_test.get_num_samples()
                    fake_size = int(np.ceil(nb_test * 1.0/self.nclasses))
                    sampled_labels = self._biased_sample_labels(nb_test, "d")
                    latent_gen = self.generate_latent(sampled_labels, bg_test)
                
                    # sample some labels from p_c and generate images from them
                    generated_images = self.generator.predict(
                        latent_gen, verbose=False)
                
                    X = np.concatenate( (bg_test.dataset_x, generated_images) )
                    aux_y = np.concatenate((bg_test.dataset_y, np.full(len(sampled_labels), self.nclasses )), axis=0)
                
                    # see if the discriminator can figure itself out...
                    test_disc_loss, test_disc_acc = self.discriminator.evaluate(
                        X, aux_y, verbose=False)
                
                    # make new latent
                    sampled_labels = self._biased_sample_labels(fake_size + nb_test, "g")
                    latent_gen = self.generate_latent(sampled_labels, bg_test)

                    test_gen_loss, test_gen_acc = self.combined.evaluate(
                        latent_gen,
                        sampled_labels, verbose=False)

                # generate an epoch report on performance
                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                # self.test_history['disc_loss'].append(test_disc_loss)
                # self.test_history['gen_loss'].append(test_gen_loss)
                # accuracy
                self.train_history['disc_acc'].append(train_disc_acc)
                self.train_history['gen_acc'].append(train_gen_acc)
                # self.test_history['disc_acc'].append(test_disc_acc)
                # self.test_history['gen_acc'].append(test_gen_acc)
                print("D_loss {}, G_loss {}, D_acc {}, G_acc {} - {}".format(
                    train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc,
                    datetime.datetime.now() - start_time
                ))
                # self.plot_his()

                # Save sample images
                if e % 15 == 0:
                    img_samples = np.array([
                        self.generate_samples(c, 10, bg_train)
                        for c in range(0,self.nclasses)
                    ])

                    utils.show_samples(img_samples)

                # Generate whole evaluation plot (real img, autoencoded img, fake img)
                if e % 10 == 5:
                    self.plot_loss_his()
                    self.plot_acc_his()
                    self.backup_point(e)
                    crt_c = 0
                    sample_size = 700
                    labels = np.zeros(sample_size)
                    img_samples = self.generate_samples(crt_c, sample_size, bg_train)
                    five_imgs = img_samples[:5]
                    for crt_c in range(1, 3):
                        new_samples = self.generate_samples(crt_c, sample_size, bg_train)
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)
                        labels = np.concatenate((np.ones(sample_size), labels), axis=0)
                        five_imgs = np.concatenate((five_imgs, new_samples[:5]), axis=0)
                    
                    labels = np_utils.to_categorical(labels, self.nclasses)
                    img_samples = np.transpose(img_samples, axes=(0, 2, 3, 1))

                    # shape = img_samples.shape
                    # img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))
                    utils.show_samples(five_imgs)
            self.trained = True

    def generate_samples(self, c, samples, bg = None):
        return self.generate(np.full(samples, c), bg)
