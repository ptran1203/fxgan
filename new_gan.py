
import csv
from collections import defaultdict
import keras.backend as K
import tensorflow as tf

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (
    UpSampling2D, Convolution2D,
    Conv2D, Conv2DTranspose
)
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Embedding, Dropout,
    BatchNormalization, Activation,
    Lambda,Layer, Add, Concatenate,
    Average,GaussianNoise,
    MaxPooling2D, AveragePooling2D
)

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

from google.colab.patches import cv2_imshow
from PIL import Image

DS_DIR = '/content/drive/My Drive/bagan/dataset/chest_xray'
DS_SAVE_DIR = '/content/drive/My Drive/bagan/dataset/save'
CLASSIFIER_DIR = '/content/drive/My Drive/chestxray_classifier'


def save_image_array(img_array, fname=None, show=None):
        # convert 1 channel to 3 channels
        print(img_array.shape)
        channels = img_array.shape[-1]
        resolution = img_array.shape[2]
        img_rows = img_array.shape[0]
        img_cols = img_array.shape[1]

        img = np.full([resolution * img_rows, resolution * img_cols, channels], 0.0)
        for r in range(img_rows):
            for c in range(img_cols):
                img[
                (resolution * r): (resolution * (r + 1)),
                (resolution * (c % 10)): (resolution * ((c % 10) + 1)),
                :] = img_array[r, c]

        img = (img * 127.5 + 127.5).astype(np.uint8)
        if show:
            try:
                cv2_imshow(img)
            except Exception as e:
                fname = '/content/drive/My Drive/bagan/result/model_{}/img_{}.png'.format(
                    resolution,
                    datetime.datetime.now().strftime("%m/%d/%Y-%H%M%S")
                )
                print('[show fail] ', str(e))
        if fname:
            try:
                Image.fromarray(img).save(fname)
            except Exception as e:
                print('Save image failed', str(e))


def show_samples(img_array):
    shape = img_array.shape
    img_samples = img_array.reshape(
        (-1, shape[-4], shape[-3], shape[-2], shape[-1])
    )
    save_image_array(img_samples, None, True)


def load_classifier(rst=256):
    json_file = open(CLASSIFIER_DIR + '/{}/model.json'.format(rst), 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    # load weights into new model
    model.load_weights(CLASSIFIER_DIR + '/{}/weights.h5'.format(rst))
    return model

def pickle_save(object, path):
    try:
        print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))



def pickle_load(path):
    try:
        print('load data from {} successfully'.format(path))
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(str(e))
        return None

def add_padding(img):
    w, h, _ = img.shape
    size = abs(w - h) // 2
    value= [0, 0, 0]
    if w < h:
        return cv2.copyMakeBorder(img, size, size, 0, 0,
                                    cv2.BORDER_CONSTANT,
                                    value=value)
    return cv2.copyMakeBorder(img, 0, 0, size, size,
                                    cv2.BORDER_CONSTANT,
                                    value=value)

def save_ds(imgs, rst, opt):
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    pickle_save(imgs, path)

def load_ds(rst, opt):
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    return pickle_load(path)

def get_img(path, rst):
    img = cv2.imread(path)
    img = add_padding(img)
    img = cv2.resize(img, (rst, rst))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(img, axis=0)
    return img.tolist()

def bound(list, s):
    if s == 0:
        return list
    return list[:s]

def load_train_data(resolution=52):
    labels = []
    i = 0
    res = load_ds(resolution, 'train')
    if res:
        return res

    files =  os.listdir(DS_DIR + '/train/NORMAL')
    imgs = np.array(get_img(DS_DIR + '/train/NORMAL/' + files[0], resolution))
    for file in files[1:]:
        path = DS_DIR + '/train/NORMAL/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs = np.concatenate((imgs, get_img(path, resolution)))
            labels.append(0)
        except:
            pass

    files =  os.listdir(DS_DIR + '/train/PNEUMONIA')
    imgs = np.concatenate((imgs,get_img(DS_DIR + '/train/PNEUMONIA/' + files[0], resolution)))
    for file in files[1:]:
        path = DS_DIR + '/train/PNEUMONIA/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs = np.concatenate((imgs, get_img(path, resolution)))
            labels.append(1)
        except:
            pass

    res = (np.array(imgs), np.array(labels))
    save_ds(res, resolution, 'train')
    return res

def load_test_data(resolution = 52):
    imgs = []
    labels = []
    res = load_ds(resolution, 'test')
    if res:
        return res
    for file in os.listdir(DS_DIR + '/test/NORMAL'):
        path = DS_DIR + '/test/NORMAL/' + file
        try:
            imgs.append(get_img(path, resolution))
            labels.append(0)
        except:
            pass

    for file in os.listdir(DS_DIR + '/test/PNEUMONIA'):
        path = DS_DIR + '/test/PNEUMONIA/' + file
        try:
            imgs.append(get_img(path, resolution))
            labels.append(1)
        except:
            pass
    res = (np.array(imgs), np.array(labels))
    save_ds(res, resolution, 'test')
    return res

class BatchGenerator:
    TRAIN = 1
    TEST = 0

    def __init__(
        self,
        data_src,
        batch_size=5,
        dataset='MNIST',
        rst=64,
        prune_classes=None,
    ):
        self.batch_size = batch_size
        self.data_src = data_src
        if self.data_src == self.TEST:
            x, y = load_test_data(rst)
            self.dataset_x = x
            self.dataset_y = y
        else:
            x, y = load_train_data(rst)
            self.dataset_x = x
            self.dataset_y = y

        # Arrange x: channel first
        self.dataset_x = np.transpose(self.dataset_x, axes=(0, 1, 2))
        # Normalize between -1 and 1
        self.dataset_x = (self.dataset_x - 127.5) / 127.5
        self.dataset_x = np.expand_dims(self.dataset_x, axis = -1)

        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        # Prune
        if prune_classes:
            for class_to_prune in range(len(classes)):
                remove_size = prune_classes[class_to_prune]
                all_ids = list(np.arange(len(self.dataset_x)))
                mask = [lc == class_to_prune for lc in self.dataset_y]
                all_ids_c = np.array(all_ids)[mask]
                np.random.shuffle(all_ids_c)
                to_delete  = all_ids_c[:remove_size]
                self.dataset_x = np.delete(self.dataset_x, to_delete, axis=0)
                self.dataset_y = np.delete(self.dataset_y, to_delete, axis=0)
                print('Remove {} items in class {}'.format(remove_size, class_to_prune))

        # Recount after pruning
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))
        self.per_class_count = per_class_count

        # List of labels
        self.label_table = [str(c) for c in range(len(self.classes))]

        # Preload all the labels.
        self.labels = self.dataset_y[:]

        # per class ids
        self.per_class_ids = dict()
        ids = np.array(range(len(self.dataset_x)))
        for c in classes:
            self.per_class_ids[c] = ids[self.labels == c]

    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size

        np.random.shuffle(self.per_class_ids[c])
        to_return = self.per_class_ids[c][0:samples]
        return self.dataset_x[to_return]

    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len( self.label_table )

    def get_class_probability(self):
        return self.per_class_count/sum(self.per_class_count)

    ### ACCESS DATA AND SHAPES ###
    def get_num_samples(self):
        return self.dataset_x.shape[0]

    def get_image_shape(self):
        return [self.dataset_x.shape[1], self.dataset_x.shape[2], self.dataset_x.shape[3]]

    def next_batch(self):
        dataset_x = self.dataset_x
        labels = self.labels

        indices = np.arange(dataset_x.shape[0])

        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern = sorted(access_pattern)

            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]

class BalancingGAN:
    def build_res_unet(self, img_dim=(64, 64, 1)):
        input_layer = Input(shape=img_dim, name="unet_input")
        stride = 2
        # 1 encoder C64
        # skip batchnorm on this layer on purpose (from paper)
        en_1 = Conv2D(kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same")(input_layer)
        en_1 = LeakyReLU(alpha=0.2)(en_1)

        # 2 encoder C128
        en_2 = Conv2D(kernel_size=(4, 4), filters=128, strides=(2, 2), padding="same")(en_1)
        en_2 = BatchNormalization(name='gen_en_bn_2')(en_2)
        en_2 = LeakyReLU(alpha=0.2)(en_2)

        # 3 encoder C256
        en_3 = Conv2D(kernel_size=(4, 4), filters=256, strides=(2, 2), padding="same")(en_2)
        en_3 = BatchNormalization(name='gen_en_bn_3')(en_3)
        en_3 = LeakyReLU(alpha=0.2)(en_3)

        # 4 encoder C512
        en_4 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_3)
        en_4 = BatchNormalization(name='gen_en_bn_4')(en_4)
        en_4 = LeakyReLU(alpha=0.2)(en_4)

        # 5 encoder C512
        en_5 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_4)
        en_5 = BatchNormalization(name='gen_en_bn_5')(en_5)
        en_5 = LeakyReLU(alpha=0.2)(en_5)

        # 6 encoder C512
        en_6 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_5)
        en_6 = BatchNormalization(name='gen_en_bn_6')(en_6)
        en_6 = LeakyReLU(alpha=0.2)(en_6)

        # 7 encoder C512
        en_7 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_6)
        en_7 = BatchNormalization(name='gen_en_bn_7')(en_7)
        en_7 = LeakyReLU(alpha=0.2)(en_7)

        # 8 encoder C512
        en_8 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_7)
        en_8 = BatchNormalization(name='gen_en_bn_8')(en_8)
        en_8 = LeakyReLU(alpha=0.2)(en_8)

        # -------------------------------
        # DECODER
        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        # 1 layer block = Conv - Upsample - BN - DO - Relu
        # also adds skip connections (Concatenate()). Takes input from previous layer matching encoder layer
        # -------------------------------
        # 1 decoder CD512 (decodes en_8)
        de_1 = Conv2DTranspose(kernel_size=(4, 4), strides = 2, filters=512, padding="same")(en_8)
        de_1 = BatchNormalization(name='gen_de_bn_1')(de_1)
        de_1 = Dropout(p=0.3)(de_1)
        de_1 = Concatenate()([de_1, en_5])
        de_1 = Activation('relu')(de_1)

        de_2 = Conv2DTranspose(kernel_size=(4, 4),strides = 2, filters=512, padding="same")(de_1)
        de_2 = BatchNormalization(name='gen_de_bn_2')(de_2)
        de_2 = Dropout(p=0.3)(de_2)
        de_2 = Concatenate()([de_2, en_4])
        de_2 = Activation('relu')(de_2)

        de_3 = Conv2DTranspose(kernel_size=(4, 4), strides = 2, filters=512, padding="same")(de_2)
        de_3 = BatchNormalization(name='gen_de_bn_3')(de_3)
        de_3 = Dropout(p=0.3)(de_3)
        de_3 = Concatenate()([de_3, en_3])
        de_3 = Activation('relu')(de_3)

        de_4 = Conv2DTranspose(kernel_size=(4, 4), strides = 2, filters=128, padding="same")(de_3)
        de_4 = BatchNormalization(name='gen_de_bn_4')(de_4)
        de_4 = Dropout(p=0.3)(de_4)
        de_4 = Concatenate()([de_4, en_2])
        de_4 = Activation('relu')(de_4)

        de_5 = Conv2DTranspose(kernel_size=(4, 4), strides = 2, filters=64, padding="same")(de_4)
        de_5 = BatchNormalization(name='gen_de_bn_5')(de_5)
        de_5 = Dropout(p=0.3)(de_5)
        de_5 = Concatenate()([de_5, en_1])
        de_5 = Activation('relu')(de_5)

        de_6 = Conv2DTranspose(kernel_size=(4, 4), strides = 2, filters=1, padding="same")(de_5)
        # de_6 = BatchNormalization(name='gen_de_bn_6')(de_6)
        # de_6 = Dropout(p=0.3)(de_6)
        # de_6 = Concatenate()([de_6, en_2])
        de_6 = Activation('tanh')(de_6)

        self.generator = Model(inputs = input_layer, outputs = de_6, name='unet_generator')



    def plot_loss_his(self):
        def toarray(lis, k):
            return [d[k] for d in lis]

        def plot_g(train_g, test_g):
            plt.plot(toarray(train_g, 'loss'), label='train_g_loss')
            plt.plot(toarray(train_g, 'loss_from_d'), label='train_g_loss_from_d')
            plt.plot(toarray(train_g, 'fm_loss'), label='train_g_loss_fm')
            plt.plot(toarray(test_g, 'loss'), label='test_g_loss')
            plt.plot(toarray(test_g, 'loss_from_d'), label='test_g_loss_from_d')
            plt.plot(toarray(test_g, 'fm_loss'), label='test_g_loss_fm')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

        def plot_d(train_d, test_d):
            plt.plot(train_d, label='train_d_loss')
            plt.plot(test_d, label='test_d_loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
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


    def plot_acc_his(self):
        def toarray(lis, k):
            return [d[k] for d in lis]

        def plot_g(train_g, test_g):
            plt.plot(train_g, label='train_g_acc')
            plt.plot(test_g, label='test_g_acc')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()
        
        def plot_d(train_d, test_d):
            plt.plot(train_d, label='train_d_acc')
            plt.plot(test_d, label='test_d_acc')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

        train_d = self.train_history['disc_acc']
        train_g = self.train_history['gen_acc']
        test_d = self.test_history['disc_acc']
        test_g = self.test_history['gen_acc']
        if len(train_g) == 0:
            return

        print(train_g)
        print(test_g)
 
        plot_g(train_g, test_g)
        plot_d(train_d, test_d)

    
    def plot_classifier_acc(self):
        plt.plot(self.classifier_acc, label='classifier_acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch(x5)')
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
        i = 0
        while crt_res < resolution/2:
            i += 1
            cnn.add(Conv2DTranspose(
                init_channels, kernel_size = 5, strides = 2, padding='same'))
            # cnn.add(BatchNormalization())
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
        self.generator = Model(inputs=latent, outputs=fake_image_from_latent, name = 'Generator')

    def _build_common_encoder(self, image, min_latent_res):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2),
        input_shape=(resolution, resolution,channels)))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        size = 128
        cnn.add(Conv2D(size, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(512, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        features = cnn(image)
        return features



    # latent_size is the innermost latent vector size; min_latent_res is latent resolution (before the dense layer).
    def build_reconstructor(self, latent_size, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution,channels))
        features = self._build_common_encoder(image, min_latent_res)
        # Reconstructor specific
        latent = Dense(latent_size, activation='linear')(features)
        self.reconstructor = Model(inputs=image, outputs=latent, name='decoder')

    def build_discriminator(self, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution,channels))
        features = self._build_common_encoder(image, min_latent_res)
        # Discriminator specific
        features = Dropout(0.4)(features)
        aux = Dense(
            self.nclasses+1, activation='softmax', name='auxiliary'  # nclasses+1. The last class is: FAKE
        )(features)
        self.discriminator = Model(inputs=image, outputs=aux,name='discriminator')


    def generate_from_latent(self, latent):
        res = self.generator(latent)
        return res

    def generate(self, c, bg=None):  # c is a vector of classes
        latent = self.generate_latent(c, bg)
        res = self.generator.predict(latent)
        return res

    def generate_latent(self, c, bg=None, n_mix=10):  # c is a vector of classes
        noise = np.random.normal(0, 0.01, self.latent_size)
        res = np.array([
            np.random.multivariate_normal(self.means[e], self.covariances[e]) + noise
            for e in c
        ])

        return res

    def discriminate(self, image):
        return self.discriminator(image)

    def features_from_d(self, image):
        return self.features_from_d_model(image)

    def build_latent_encoder(self):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution,channels))
        features = self._build_common_encoder(image, self.min_latent_res)
        # Reconstructor specific
        latent = Dense(100, activation='linear')(features)
        self.latent_encoder = Model(inputs=image, outputs=latent)

    def discriminator_feature_layer(self):
        return self.discriminator.layers[-3]

    def build_features_from_d_model(self):
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        model_output = self.discriminator.layers[-3](image)
        self.features_from_d_model = Model(
            inputs = image,
            output = model_output,
            name = 'Feature_matching'
        )


    def build_g_trigger(self):
            self.build_res_unet()
            # self.build_generator(self.latent_size, self.min_latent_res)

    def __init__(self, classes, target_class_id,
                # Set dratio_mode, and gratio_mode to 'rebalance' to bias the sampling toward the minority class
                # No relevant difference noted
                dratio_mode="uniform", gratio_mode="uniform",
                adam_lr=0.00005, latent_size=100,
                res_dir = "./res-tmp", image_shape=[3,32,32], min_latent_res=8,
                g_lr = 0.000005):
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.classes = classes
        self.target_class_id = target_class_id  # target_class_id is used only during saving, not to overwrite other class results.
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[0]
        self.g_lr = g_lr

        self.min_latent_res = min_latent_res
        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build generator
        # self.build_generator(latent_size, init_resolution=min_latent_res)
        self.build_g_trigger()

        latent_gen = Input(shape=(latent_size, ))
        real_images = Input(shape=(self.resolution, self.resolution, self.channels))

        # Build discriminator
        self.build_discriminator(min_latent_res=min_latent_res)
        self.discriminator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics=['accuracy'],
            loss='sparse_categorical_crossentropy'
        )

        # Build reconstructor
        self.build_reconstructor(latent_size, min_latent_res=min_latent_res)

        # Define combined for training generator.
        fake = self.generator(real_images)

        self.build_features_from_d_model()

        self.discriminator.trainable = False
        self.reconstructor.trainable = False
        self.generator.trainable = True
        self.features_from_d_model.trainable = False
        aux = self.discriminate(fake)

        fake_features = self.features_from_d(fake)

        self.combined = Model(
            inputs=real_images,
            outputs=[aux, fake_features],
            name = 'Combined'
        )

        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            metrics=['accuracy'],
            loss= ['sparse_categorical_crossentropy', 'mse'],
            # loss_weights = [1.0, 0.0],
        )

        # Define initializer for autoencoder
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.reconstructor.trainable = True

        img_for_reconstructor = Input(shape=(self.resolution, self.resolution,self.channels))

        img_reconstruct = self.generator(img_for_reconstructor)
        self.autoenc_0 = Model(
            inputs=img_for_reconstructor,
            outputs=img_reconstruct,
            name = 'autoencoder'
        )
        self.autoenc_0.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

    def _biased_sample_labels(self, samples, target_distribution="uniform"):
        all_labels = np.full(samples, 0)
        splited = np.array_split(all_labels, self.nclasses)
        all_labels = np.concatenate(
            [
                np.full(splited[classid].shape[0], classid) \
                for classid in range(self.nclasses)
            ]
        )
        np.random.shuffle(all_labels)
        return all_labels

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
            generated_images = self.generator.predict(
                image_batch, verbose=0
            )
    
            X = np.concatenate((image_batch, generated_images))
            aux_y = np.concatenate((label_batch, np.full(generated_images.shape[0] , self.nclasses )), axis=0)
            
            X, aux_y = self.shuffle_data(X, aux_y)
            loss, acc = self.discriminator.train_on_batch(X, aux_y)
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################
            real_features = self.features_from_d_model.predict(image_batch)
            # ['loss', 'discriminator_loss', 'Feature_matching_loss',
            #   'discriminator_accuracy', 'Feature_matching_accuracy']
            [
                loss, discriminator_loss,
                feature_matching_loss,
                discriminator_accuracy,
                feature_matching_accuracy
            ] = self.combined.train_on_batch(
                image_batch,
                [label_batch, real_features]
            )

            epoch_gen_loss.append({
                'loss': loss,
                'loss_from_d': discriminator_loss,
                'fm_loss': feature_matching_loss
            })
            epoch_gen_acc.append(discriminator_accuracy)

        epoch_gen_loss_cal = {
            'loss': np.mean(np.array([e['loss'] for e in epoch_gen_loss])),
            'loss_from_d': np.mean(np.array([e['loss_from_d'] for e in epoch_gen_loss])),
            'fm_loss': np.mean(np.array([e['fm_loss'] for e in epoch_gen_loss]))
        }

        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            epoch_gen_loss_cal,
            np.mean(np.array(epoch_disc_acc), axis=0),
            np.mean(np.array(epoch_gen_acc), axis=0),
        )

    def shuffle_data(self, data_x, data_y):
        rd_idx = np.arange(data_x.shape[0])
        np.random.shuffle(rd_idx)
        return data_x[rd_idx], data_y[rd_idx]

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
            self.autoenc_epochs = 100
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
            if re.match(r'bck_c_{}'.format(self.target_class_id) + "_" + element, f)
        ]
        if len(files) > 0:
            fname = files[0]
            e_str = os.path.splitext(fname)[0].split("_")[-1]

            epoch = int(e_str)

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
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            return epoch

        # Return epoch
        except Exception as e:  # Reload error, restart from scratch (the first time we train we pass from here)
            print(str(e))
            return 0

    def backup_point(self, epoch):
        # Remove last bck
        _, old_bck_g = self._get_lst_bck_name("generator")
        _, old_bck_d = self._get_lst_bck_name("discriminator")
        try:
            os.remove(os.path.join(self.res_dir, old_bck_g))
            os.remove(os.path.join(self.res_dir, old_bck_d))
        except:
            pass

        # Bck
        generator_fname = "{}/bck_c_{}_generator_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)
        discriminator_fname = "{}/bck_c_{}_discriminator_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)

        self.generator.save(generator_fname)
        self.discriminator.save(discriminator_fname)
        # pickle_save(self.classifier_acc, CLASSIFIER_DIR + '/acc_array.pkl')

    def evaluate_d(self, test_x, test_y):
        y_pre = self.discriminator.predict(test_x)
        y_pre = np.argmax(y_pre, axis=1)
        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)  # shape=(12, 12)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def evaluate_g(self, test_x, test_y):
        y_pre, _ = self.combined.predict(test_x)
        y_pre = np.argmax(y_pre, axis=1)
        cm = metrics.confusion_matrix(y_true=test_y[0], y_pred=y_pre)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            self.autoenc_epochs = 100

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
                        act_img_samples
                    ),
                ]
            ])
            for crt_c in range(1, self.nclasses):
                act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                new_samples = np.array([
                    [
                        act_img_samples,
                        self.generator.predict(
                            act_img_samples
                        ),
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            show_samples(img_samples)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc = self._train_one_epoch(bg_train)

                # Test: # generate a new batch of noise
                nb_test = bg_test.get_num_samples()
            
                # sample some labels from p_c and generate images from them
                generated_images = self.generator.predict(
                    bg_test.dataset_x, verbose=False
                )

                X = np.concatenate( (bg_test.dataset_x, generated_images) )
                aux_y = np.concatenate((bg_test.dataset_y, np.full(
                    generated_images.shape[0], self.nclasses )), axis=0
                )

                # see if the discriminator can figure itself out...
                test_disc_loss, test_disc_acc = self.discriminator.evaluate(
                    X, aux_y, verbose=False)

                real_features = self.features_from_d_model.predict(bg_test.dataset_x)
                [
                    loss, discriminator_loss,
                    feature_matching_loss,
                    discriminator_accuracy,
                    feature_matching_accuracy
                ] = self.combined.evaluate(
                    bg_test.dataset_x,
                    [bg_test.dataset_y, real_features]
                )

                if e % 25 == 0:
                    self.evaluate_d(X, aux_y)
                    self.evaluate_g(
                        bg_test.dataset_x,
                        [bg_test.dataset_y, real_features]
                    )

                    crt_c = 0
                    act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    img_samples = np.array([
                        [
                            act_img_samples,
                            self.generator.predict(
                                act_img_samples
                            ),
                        ]
                    ])
                    for crt_c in range(1, self.nclasses):
                        act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        new_samples = np.array([
                            [
                                act_img_samples,
                                self.generator.predict(
                                    act_img_samples
                                ),
                            ]
                        ])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)

                    show_samples(img_samples)

                    self.plot_loss_his()
                    self.plot_acc_his()

                if e % 100 == 0:
                    self.backup_point(e)

                self.interval_process(e)


                print("D_loss {}, G_loss {}, D_acc {}, G_acc {} - {}".format(
                    train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc,
                    datetime.datetime.now() - start_time
                ))
                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append({
                    'loss': loss,
                    'loss_from_d': discriminator_loss,
                    'fm_loss': feature_matching_loss
                })
                # accuracy
                self.train_history['disc_acc'].append(train_disc_acc)
                self.train_history['gen_acc'].append(train_gen_acc)
                self.test_history['disc_acc'].append(test_disc_acc)
                self.test_history['gen_acc'].append(discriminator_accuracy)
                # self.plot_his()

            self.trained = True

    def generate_samples(self, c, samples, bg = None):
        return self.generate(np.full(samples, c), bg)
    
    def interval_process(self, epoch, interval = 20):
        if epoch % interval != 0:
            return
        # do bussiness thing

    def save_history(self, res_dir, class_id):
        if self.trained:
            filename = "{}/class_{}_score.csv".format(res_dir, class_id)
            generator_fname = "{}/class_{}_generator.h5".format(res_dir, class_id)
            discriminator_fname = "{}/class_{}_discriminator.h5".format(res_dir, class_id)
            reconstructor_fname = "{}/class_{}_reconstructor.h5".format(res_dir, class_id)
            with open(filename, 'w') as csvfile:
                fieldnames = [
                    'train_gen_loss', 'train_disc_loss',
                    'test_gen_loss', 'test_disc_loss'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for e in range(len(self.train_history['gen_loss'])):
                    row = [
                        self.train_history['gen_loss'][e],
                        self.train_history['disc_loss'][e],
                        self.test_history['gen_loss'][e],
                        self.test_history['disc_loss'][e]
                    ]

                    writer.writerow(dict(zip(fieldnames,row)))

            self.generator.save(generator_fname)
            self.discriminator.save(discriminator_fname)
            self.reconstructor.save(reconstructor_fname)

    def load_models(self, fname_generator, fname_discriminator, fname_reconstructor, bg_train=None):
        self.init_autoenc(bg_train, gen_fname=fname_generator, rec_fname=fname_reconstructor)
        self.discriminator.load_weights(fname_discriminator)