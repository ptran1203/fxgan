
import csv
from collections import defaultdict, Counter
import keras.backend as K
import tensorflow as tf
import keras
from tensorflow.examples.tutorials.mnist import input_data

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
    RepeatVector,
)
from keras_contrib.losses import DSSIMObjective

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


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

from google.colab.patches import cv2_imshow
from PIL import Image

DS_DIR = '/content/drive/My Drive/bagan/dataset/chest_xray'
DS_DIR_2 = '/content/drive/My Drive/bagan/dataset/multi_chest'
DS_SAVE_DIR = '/content/drive/My Drive/bagan/dataset/save'
CLASSIFIER_DIR = '/content/drive/My Drive/chestxray_classifier'

# ===========================================
# I have no idea how it works, happy coding #
#               ¯\_(ツ)_/¯                  #
#===========================================#

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))


def save_image_array(img_array, fname=None, show=None):
        # convert 1 channel to 3 channels
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

def triple_channels(image):
    # axis = 2 for single image, 3 for many images
    return np.repeat(image, 3, axis = -1)


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
    """
    Add black padding
    """
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
    # path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    path = '{}/imgs_{}.pickle'.format(DS_DIR_2, rst)
    return pickle_load(path)

def get_img(path, rst):
    img = cv2.imread(path)
    img = add_padding(img)
    img = cv2.resize(img, (rst, rst))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(img, axis=0)
    return img.tolist()

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


def pred2bin(pred):
    """
    Convert probability prediction of sigmoid into binary
    """
    for x in pred:
        if x[0] >= 0.5:
            x[0] = 1
        else:
            x[0] = 0
    return pred



class BatchGenerator:
    TRAIN = 1
    TEST = 0
    D_SIZE = 400

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
        if dataset == 'MNIST':
            mnist = input_data.read_data_sets("dataset/mnist", one_hot=False)

            if self.data_src == self.TEST:
                self.dataset_x = mnist.test.images
                self.dataset_y = mnist.test.labels
            else:
                self.dataset_x = mnist.train.images
                self.dataset_y = mnist.train.labels

            # Normalize between -1 and 1
            self.dataset_x = self.dataset_x.reshape((self.dataset_x.shape[0], 28, 28, 1))
            self.dataset_x = (self.dataset_x * 255.0 - 127.5) / 127.5

            # Include 1 single color channel
            self.dataset_x = np.expand_dims(self.dataset_x, axis=-1)

        elif dataset == 'chest':
            if self.data_src == self.TEST:
                x, y = load_test_data(rst)
                self.dataset_x = x
                self.dataset_y = y

            else:
                x, y = load_train_data(rst)
                self.dataset_x = x
                self.dataset_y = y

        else: # multi chest
            x, y = pickle_load('/content/drive/My Drive/bagan/dataset/multi_chest/imgs_labels.pkl')
            
            if self.data_src == self.TEST:
                self.dataset_x = x[:self.D_SIZE]
                # TODO: HARD CODE HERE
                self.dataset_y = np.ones((self.dataset_x.shape[0], 1))
            else:
                self.dataset_x = x[self.D_SIZE:self.D_SIZE * 2]
                # TODO: HARD CODE HERE
                self.dataset_y = np.ones((self.dataset_x.shape[0], 1))



        # Arrange x: channel first
        if self.data_src == self.TEST and rst == 32:
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 2, 3, 1))
        # Normalize between -1 and 1
        self.dataset_x = (self.dataset_x - 127.5) / 127.5
        if self.data_src == self.TRAIN or rst == 64:
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
        try:
            for c in classes:
                self.per_class_ids[c] = ids[self.labels == c]
        except:
            pass

    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size
        try:
            np.random.shuffle(self.per_class_ids[c])
            to_return = self.per_class_ids[c][0:samples]
            return self.dataset_x[to_return]
        except:
            random = np.arange(self.dataset_x.shape[0])
            np.random.shuffle(random)
            to_return = random[:samples]
            return self.dataset_x[to_return]

    def get_samples_by_labels(self, labels, samples = None):
        if samples is None:
            samples = self.batch_size

        count = Counter(labels)
        class_1 = np.random.choice(self.per_class_ids[0], count[0])
        class_2 = np.random.choice(self.per_class_ids[1], count[1])
        new_arr = []

        for label in labels:
            if label == 0:
                idx, class_1 = class_1[-1], class_1[:-1]
            else:
                idx, class_2 = class_2[-1], class_2[:-1]
            new_arr.append(idx)

        return self.dataset_x[np.array(new_arr)]

        

    def pair_samples(self, train_x):
        # merge 2 nearest image
        img1 = np.expand_dims(train_x[0], 0)
        img2 = np.expand_dims(train_x[1], 0)
        pair_x = np.array([np.concatenate((img1, img2))])
        for i in range(2, len(train_x) - 1, 2):
            img1 = np.expand_dims(train_x[i], 0)
            img2 = np.expand_dims(train_x[i + 1], 0)
            pair_x = np.concatenate((pair_x, np.expand_dims(
                                    np.concatenate((img1, img2)), 0)))

        return pair_x

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
        indices2 = np.arange(dataset_x.shape[0])

        np.random.shuffle(indices)
        np.random.shuffle(indices2)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern2 = indices2[start_idx:start_idx + self.batch_size]

            yield (
                dataset_x[access_pattern, :, :, :], labels[access_pattern],
                dataset_x[access_pattern2, :, :, :], labels[access_pattern2]
            )

    def next_pair_batch(self):
        dataset_x = self.pair_x
        labels = self.pair_y

        indices = np.arange(dataset_x.shape[0])

        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern = sorted(access_pattern)

            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]


class RandomPick(keras.layers.Layer):
    def __init__(self):
        super(RandomPick, self).__init__()

    def call(self, inputs):
        ip1, ip2, vector = inputs
        out = []

        for i in range(ip1.shape[-1]):
            r = tf.cond(vector[0,i] >= -5, lambda: ip1[:, :, :, i], lambda: ip2[:, :, :, i])
            # merged = vector[0, i] * ip1[]
            out.append(r)

        return tf.transpose(tf.stack(out), [1, 2, 3, 0])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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
    def __init__(self, epsilon = 1e-6, norm = 'batch'):
        super(FeatureNorm, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def call(self, inputs):
        x, scale, bias = inputs

        # x = [batch, height, width, channels]
        axis = [-1] # instance norm
        if self.norm == 'batch':
            axis = [0]
        axis = [1, 2]

        mean = K.mean(x, axis = axis, keepdims = True)
        std = K.std(x, axis = axis, keepdims = True)
        norm = (x - mean) * (1 / (std + self.epsilon))

        broadcast_scale = K.reshape(scale, (-1, 1, 1, 1))
        broadcast_bias = K.reshape(bias, (-1, 1, 1, 1))

        return norm * broadcast_scale + broadcast_bias

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class BalancingGAN:
    def _res_block(self,  x, activation = 'leaky_relu', norm = 'batch', scale=None, bias=None):
        if activation == 'leaky_relu':
            actv = LeakyReLU()
        else:
            actv = Activation(activation)

        def norm_layer(x):
            if norm == 'batch':
                x = BatchNormalization()(x)
            else:
                x = FeatureNorm()([x, scale, bias])
            return x

        skip = Conv2D(64, 3, strides = 1, padding = 'same')(x)
        skip = norm_layer(skip)
        out = actv(skip)

        skip = Conv2D(64, 1, strides = 1, padding = 'same')(skip)

        out = Conv2D(64, 3, strides = 1, padding = 'same')(out)
        out = norm_layer(out)
        out = actv(out)

        out = Conv2D(64, 3, strides = 1, padding = 'same')(out)
        out = norm_layer(out)
        out = actv(out)
        out = Add()([out, skip])
        return out

    def build_latent_encoder(self):
        """
        Mapping image to latent code
        """
        image = Input(shape=(self.resolution, self.resolution, self.channels))

        x = self._res_block(image, 'relu')
        x = Conv2D(32, 3, strides = 2, padding = 'same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        # 32 * 32 * 128

        # x = self._res_block(x, 'relu')
        x = Conv2D(64, 3, strides = 2, padding = 'same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        # 16 * 16 * 64

        # x = self._res_block(x, 'relu')
        x = Conv2D(64, 3, strides = 2, padding = 'same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        # 8*8*64

        # x = self._res_block(x, 'relu')
        x = Conv2D(128, 3, strides = 2, padding = 'same')(x)
        x = self._norm()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        # 4*4*32

        x = AveragePooling2D()(x)

        latent = Flatten()(x)
        latent = Dense(512, activation = 'relu')(latent)

        self.latent_encoder = Model(
            inputs = image,
            outputs = latent    
        )

    def __init__(self, classes, loss_type = 'binary',
                adam_lr=0.00005, latent_size=100,
                res_dir = "./res-tmp", image_shape=[32, 32, 1],
                g_lr = 0.000005, norm = 'batch'):
        self.classes = classes
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[0]
        self.g_lr = g_lr

        self.norm = norm
        if loss_type == 'binary':
            print('LOSS TYPE: BinaryCrossentropy')
            self.g_loss = keras.losses.BinaryCrossentropy()
            self.d_fake_loss = keras.losses.BinaryCrossentropy()
            self.d_real_loss = keras.losses.BinaryCrossentropy()
        elif loss_type == 'hinge':
            print('LOSS TYPE: Hinge')
            self.g_loss = hinge_G_loss
            self.d_fake_loss = hinge_D_fake_loss
            self.d_real_loss = hinge_D_real_loss
        else:
            print('LOSS TYPE: wasserstein')
            self.g_loss = wasserstein_loss
            self.d_fake_loss = wasserstein_loss
            self.d_real_loss = wasserstein_loss

        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build generator
        self.build_perceptual_model()
        self.build_attribute_net()
        self.build_discriminator()
        self.build_features_from_d_model()
        self.build_latent_encoder()
        self.build_res_unet()

        real_images = Input(shape=(self.resolution, self.resolution, self.channels))
        other_batch = Input(shape=(self.resolution, self.resolution, self.channels))
        latent_code = Input(shape=(self.latent_size,))

        fake_images = Input(shape=(self.resolution, self.resolution, self.channels))
        # scale, bias = self.attribute_net(real_images)

        real_output_for_d = self.discriminator(real_images)
        fake_output_for_d = self.discriminator(fake_images)

        self.discriminator_model = Model(
            inputs = [real_images, fake_images],
            outputs = [fake_output_for_d, real_output_for_d],
        )
        self.discriminator_model.compile(
            optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics = ['accuracy'],
            loss = [self.d_fake_loss, self.d_real_loss]
        )

        # Define combined for training generator.
        fake = self.generator([
            real_images, other_batch, latent_code
        ])

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.features_from_d_model.trainable = False
        # self.latent_encoder.trainable = False

        # Model to train d
        aux_fake = self.discriminator(fake)

        self.combined = Model(
            inputs=[real_images, other_batch, latent_code],
            outputs=[aux_fake],
            name = 'Combined'
        )

        fake_perceptual_features = self.vgg16_features(fake)
        real_perceptual_features = self.vgg16_features(other_batch)

        # self.combined.add_loss(K.mean(K.abs(fake_perceptual_features - real_perceptual_features)))

        # performce triplet loss
        margin = 1.0
        d_pos = K.sum(K.square(self.vgg16_features(fake) - self.vgg16_features(other_batch)))
        d_neg = K.sum(K.square(self.vgg16_features(fake) - self.vgg16_features(real_images)))
        self.combined.add_loss(K.maximum(d_pos - d_neg + margin, 0.))


        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            metrics=['accuracy'],
            loss = self.g_loss,
        )

    def vgg16_features(self, image):
        return self.perceptual_model(Concatenate()([
            image, image, image
        ]))

    def build_attribute_net(self):
        image = Input((self.resolution, self.resolution, self.channels))
        feature = self.vgg16_features(image)
        attr_feature = Flatten()(feature)

        scale = Dense(256,)(attr_feature)
        scale = Dense(1, name = 'norm_scale')(scale)
        bias = Dense(256,)(attr_feature)
        bias = Dense(1, name = 'norm_bias')(bias)

        self.attribute_net = Model(inputs = image, outputs = [scale, bias],
                                   name = 'attribute_net')

    def build_res_unet(self):
        def _encoder(activation = 'relu'):
            if activation == 'leaky_relu':
                actv = LeakyReLU()
            else:
                actv = Activation(activation)

            image = Input(shape=(self.resolution, self.resolution, self.channels))

            en_1 = self._res_block(image, activation)
            en_1 = Conv2D(64, 5, strides=(2, 2), padding="same")(en_1)
            en_1 = self._norm()(en_1)
            en_1 = actv(en_1)
            en_1 = Dropout(0.3)(en_1)
            # out_shape: 32*32*64

            en_2 = self._res_block(en_1, activation)
            en_2 = Conv2D(64, 5, strides=(2, 2), padding="same")(en_2)
            en_2 = self._norm()(en_2)
            en_2 = actv(en_2)
            en_2 = Dropout(0.3)(en_2)
            # out_shape:  16*16*64

            en_3 = self._res_block(en_2, activation)
            en_3 = Conv2D(128, 5, strides = 2, padding = 'same')(en_3)
            en_3 = self._norm()(en_3)
            en_3 = actv(en_3)
            en_3 = Dropout(0.3)(en_3)
            # out_shape: 8*8*128

            en_4 = self._res_block(en_3, activation)
            en_4 = Conv2D(128, 5, strides = 2, padding = 'same')(en_4)
            en_4 = self._norm()(en_4)
            en_4 = actv(en_4)
            en_4 = Dropout(0.3, name = 'decoder_output')(en_4)
            # out_shape: 4 4 128

            return Model(inputs = image, outputs = [en_1, en_2, en_3, en_4])

        image = Input(shape=(self.resolution, self.resolution, self.channels), name = 'image_1')
        image2 = Input(shape=(self.resolution, self.resolution, self.channels), name = 'image_2')

        latent_code = Input(shape=(128,), name = 'latent_code')

        self.encoder = _encoder()
        feature = self.encoder(image)
        feature2 = self.encoder(image2)
        
        # scale, bias = self.attribute_net(image2)


        hw = int(0.0625 * self.resolution)
        latent_noise1 = Dense(hw*hw*128,)(latent_code)
        latent_noise1 = Reshape((hw, hw, 128))(latent_noise1)

        hw *= 2
        latent_noise2 = Dense(hw*hw*128,)(latent_code)
        latent_noise2 = Reshape((hw, hw, 128))(latent_noise2)

        hw *= 2
        latent_noise3 = Dense(hw*hw*64,)(latent_code)
        latent_noise3 = Reshape((hw, hw, 64))(latent_noise3)

        en_1 = feature[0]
        en_2 = feature[1]
        en_3 = feature[2]
        en_4 = feature[3]

        en_4 = Concatenate()([en_4, feature2[3]])
        en_3 = Concatenate()([en_3, feature2[2]])
        en_2 = Concatenate()([en_2, feature2[1]])
        en_1 = Concatenate()([en_1, feature2[0]])

        # botteneck
        decoder_activation = Activation('relu')
        # de_1 = self._res_block(en_4, activation='relu', norm = 'feature', scale=scale, bias=bias)
        de_1 = self._res_block(en_4, 'relu')
        de_1 = Conv2DTranspose(256, 5, strides = 2, padding = 'same')(de_1)
        de_1 = decoder_activation(de_1)
        # de_1 = FeatureNorm()([de_1, scale, bias])
        de_1 = self._norm()(de_1)
        de_1 = Dropout(0.3)(de_1)
        de_1 = Add()([de_1, en_3])

        # de_1 = SelfAttention(128)(de_1)

        # de_2 = self._res_block(de_1, activation='relu', norm = 'feature', scale=scale, bias=bias)
        de_2 = self._res_block(de_1, 'relu')
        de_2 = Conv2DTranspose(128, 5, strides = 2, padding = 'same')(de_2)
        de_2 = decoder_activation(de_2)
        # de_2 = FeatureNorm()([de_2, scale, bias])
        de_2 = self._norm()(de_2)
        de_2 = Dropout(0.3)(de_2)
        de_2 = Add()([de_2, en_2])

        # de_3 = self._res_block(de_2, activation='relu', norm = 'feature', scale=scale, bias=bias)
        de_3 = self._res_block(de_2, 'relu')
        de_3 = Conv2DTranspose(128, 5, strides = 2, padding = 'same')(de_3)
        de_3 = decoder_activation(de_3)
        # de_3 = FeatureNorm()([de_3, scale, bias])
        de_3 = self._norm()(de_3)
        de_3 = Dropout(0.3)(de_3)
        de_3 = Add()([de_3, en_1])

        final = Conv2DTranspose(1, 5, strides = 2, padding = 'same')(de_3)
        outputs = Activation('tanh')(final)

        self.generator = Model(
            inputs = [image, image2, latent_code],
            outputs = outputs,
            name='unet'
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

        # plot_g(train_g, test_g)
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
 
        plot_g(train_g, test_g)
        plot_d(train_d, test_d)


    def _build_common_encoder(self, image):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2)))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))

        cnn.add(Conv2D(128, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(SelfAttention(256))
        # cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(512, (5, 5), padding='same', strides=(2, 2)))
        # cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))
        # cnn.add(Dropout(0.3))

        # cnn.add(Flatten())

        features = cnn(image)
        return features

    def build_discriminator(self):
        resolution = self.resolution
        channels = self.channels

        image = Input(shape=(resolution, resolution, channels))
        # scale bias for feature norm
        scale = Input((1,))
        bias = Input((1,))

        features = self._build_common_encoder(image)
        # features = FeatureNorm()([features, scale, bias])
        features = Flatten()(features)
        features = Dropout(0.3)(features)
        aux = Dense(
            1, name='auxiliary'
        )(features)

        self.discriminator = Model(inputs=image,
                                   outputs=aux,
                                   name='discriminator')


    def generate_latent(self, c, size = 1):
        return np.array([
            np.random.normal(0, 1, self.latent_size)
            for i in c
        ])

    def build_features_from_d_model(self):
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        model_output = self.discriminator.layers[-3](image)
        self.features_from_d_model = Model(
            inputs = image,
            output = model_output,
            name = 'Feature_matching'
        )

    def _norm(self):
        return BatchNormalization() if self.norm == 'batch' else InstanceNormalization()


    def get_pair_features(self, image_batch):
        features = self.features_from_d_model.predict(image_batch)
        p_features = self.perceptual_model.predict(triple_channels(image_batch))

        return features, p_features

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        epoch_disc_acc = []
        epoch_gen_acc = []

        for image_batch, label_batch, image_batch2, label_batch2 in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################
            fake_size = crt_batch_size // self.nclasses
            f = self.generate_latent(range(image_batch.shape[0]))
            for i in range(2):
                generated_images = self.generator.predict(
                    [
                        image_batch,
                        image_batch2,
                        f,
                    ],
                    verbose=0
                )

                # X, aux_y = self.shuffle_data(X, aux_y)
                fake_label = np.ones((generated_images.shape[0], 1))
                real_label = -np.ones((label_batch.shape[0], 1))

                loss, acc, *rest = self.discriminator_model.train_on_batch(
                    [image_batch2, generated_images],
                    [fake_label, real_label]
                )
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################
            f = self.generate_latent(range(crt_batch_size))

            [loss, acc, *rest] = self.combined.train_on_batch(
                [image_batch, image_batch2, f],
                [real_label],
            )

            epoch_gen_loss.append(loss)
            epoch_gen_acc.append(acc)

        # In case generator have multiple metrics
        # epoch_gen_loss_cal = {
        #     'loss': np.mean(np.array([e['loss'] for e in epoch_gen_loss])),
        #     'loss_from_d': np.mean(np.array([e['loss_from_d'] for e in epoch_gen_loss])),
        #     'fm_loss': np.mean(np.array([e['fm_loss'] for e in epoch_gen_loss]))
        # }

        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0),
            np.mean(np.array(epoch_disc_acc), axis=0),
            np.mean(np.array(epoch_gen_acc), axis=0),
        )

    def shuffle_data(self, data_x, data_y):
        rd_idx = np.arange(data_x.shape[0])
        np.random.shuffle(rd_idx)
        return data_x[rd_idx], data_y[rd_idx]

    def _get_lst_bck_name(self, element):
        # Find last bck name
        files = [
            f for f in os.listdir(self.res_dir)
            if re.match(r'bck_' + "_" + element, f)
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
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            return epoch

        # Return epoch
        except Exception as e:  # Reload error, restart from scratch (the first time we train we pass from here)
            print(str(e))
            return 0

    def backup_point(self, epoch):
        # Bck
        print('Save weights at epochs : ', epoch)
        generator_fname = "{}/bck_generator.h5".format(self.res_dir)
        discriminator_fname = "{}/bck_discriminator.h5".format(self.res_dir)

        self.generator.save(generator_fname)
        self.discriminator.save(discriminator_fname)
        # pickle_save(self.classifier_acc, CLASSIFIER_DIR + '/acc_array.pkl')

    def evaluate_d(self, test_x, test_y):
        y_pre = self.discriminator.predict(test_x)
        if y_pre[0].shape[0] > 1:
            y_pre = np.argmax(y_pre, axis=1)
        else:
            y_pre = pred2bin(y_pre)
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
            act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            random_samples = bg_train.get_samples_for_class(crt_c, 10)
            f = self.generate_latent(range(10))
    
            img_samples = np.array([
                [
                    act_img_samples,
                    self.generator.predict([
                        act_img_samples,
                        random_samples,
                        f,
                    ]),
                ]
            ])
            for crt_c in range(1, self.nclasses):
                act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                random_samples = bg_train.get_samples_for_class(crt_c, 10)
                new_samples = np.array([
                    [
                        act_img_samples,
                        self.generator.predict([
                            act_img_samples,
                            random_samples,
                            f,
                        ]),
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            show_samples(img_samples)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc = self._train_one_epoch(bg_train)
            
                f = self.generate_latent(range(bg_test.dataset_x.shape[0]))

                generated_images = self.generator.predict(
                    [
                        bg_test.dataset_x,
                        self.shuffle_data(bg_test.dataset_x, bg_test.dataset_y)[0],
                        f
                    ],
                    verbose=False
                )

                X = np.concatenate([bg_test.dataset_x, generated_images])
    
                aux_y = np.concatenate([
                    np.full(bg_test.dataset_y.shape[0], 0),
                    np.full(generated_images.shape[0], 1)
                ])

                fake_label = np.ones((bg_test.dataset_y.shape[0], 1))
                real_label = -np.ones((generated_images.shape[0], 1))

                X = [bg_test.dataset_x, generated_images]
                Y = [fake_label, real_label]

                test_disc_loss, test_disc_acc, *rest = self.discriminator_model.evaluate(X, Y, verbose=False)

                [test_gen_loss, test_gen_acc, *rest] = self.combined.evaluate(
                    [
                        bg_test.dataset_x,
                        self.shuffle_data(bg_test.dataset_x, bg_test.dataset_y)[0],
                        f
                    ],
                    [real_label],
                    verbose = 0
                )

                if e % 25 == 0:
                    self.evaluate_d(X, Y)
                    self.evaluate_g(
                        [
                            bg_test.dataset_x,
                            self.shuffle_data(bg_test.dataset_x, bg_test.dataset_y)[0],
                            f,
                            
                        ],
                        [real_label],
                        # [bg_test.dataset_y]
                    )

                    crt_c = 0
                    act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    random_imgs = bg_train.get_samples_for_class(1, 10)

                    f = self.generate_latent(range(10))
                    img_samples = np.array([
                        [
                            act_img_samples,
                            random_imgs,
                            self.generator.predict([
                                act_img_samples,
                                random_imgs,
                                f,
                                
                            ]),
                        ]
                    ])
                    for crt_c in range(1, self.nclasses):
                        act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        random_imgs = bg_train.get_samples_for_class(0, 10)
                        f = self.generate_latent(range(10))
                        new_samples = np.array([
                            [
                                act_img_samples,
                                random_imgs,
                                self.generator.predict([
                                    act_img_samples,
                                    random_imgs,
                                    f,
                                    
                                ]),
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
                self.test_history['gen_loss'].append(test_gen_loss)
                # accuracy
                self.train_history['disc_acc'].append(train_disc_acc)
                self.train_history['gen_acc'].append(train_gen_acc)
                self.test_history['disc_acc'].append(test_disc_acc)
                self.test_history['gen_acc'].append(test_gen_acc)
                # self.plot_his()

            self.trained = True


    def generate_samples(self, c, samples, bg = None):
        """
        Refactor later
        """
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