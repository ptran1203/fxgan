%tensorflow_version 1.x

import sys
import os
import pickle
import cv2
import keras.backend as K
import tensorflow as tf
import re
import numpy as np
import keras
from google.colab import drive
from collections import defaultdict
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Embedding, Dropout,
    BatchNormalization, Activation,
    Lambda,Layer, Add, Concatenate
)
from keras.utils import np_utils, plot_model


from keras.layers import multiply as kmultiply
from keras.layers import add as kadd
import csv
from PIL import Image
from google.colab import drive

# drive.mount('/content/drive')

BASE_DIR = '/content/drive/My Drive/bagan'

def pickle_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def load_ds(rst, opt):
    path = '{}/imgs_{}_{}.pkl'.format('/content/drive/My Drive/bagan/dataset/save', opt, rst)
    return pickle_load(path)

def generate_data(dataset, n_way, k_shot):
    query_size = 15
    train_x, train_y = dataset
    idx = np.random.choice(
                train_y,
                size = (query_size,)
                )
    
    train_y = np_utils.to_categorical(train_y, n_way)
    
    query_imgs = train_x[idx], train_y[idx]
    train_x = np.delete(train_x, idx, axis = 0)
    train_y = np.delete(train_y, idx, axis = 0)

    idx0 = np.where(train_y == 0)[0]
    idx1 = np.where(train_y == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    idx0 = idx0[:k_shot]
    idx1 = idx1[:k_shot]

    c_idx = np.concatenate((idx0, idx1))

    train_x = train_x[c_idx]
    train_y = train_y[c_idx]

    return (train_x, train_y), (query_imgs)

class RelationNet():
    def __init__(self, img_rst, dataset):
        self.rst = img_rst
        self.build_model()

    def _embedding_module(self, x):
        # input_tensor = Input((img_size, img_size, 1))
        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def _relation_module(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides = 2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        # relation score
        model.add(Dense(1, activation='sigmoid', name='sigmoid_output'))
        return model

    def build_model(self):
        img_size = self.rst
        c_way = 2
        k_shot = 5
        support_size = c_way * k_shot

        img = Input((img_size, img_size, 1))
        support_img = Input((img_size, img_size, 1))

        feature = self._embedding_module()(img) 
        support_feature = self._embedding_module()(support_img)

        concat_feature = Concatenate()([feature, support_feature])

        relation_scores = self._relation_module()(concat_feature)

        self.model = Model(inputs = [imgs, support_img], outputs = relation_scores)
        self.model.compile(optimizer = 'adam', loss = 'mse')

    def plot(self):
        plot_model(self.model, to_file='/content/drive/My Drive/bagan/rn_model.png')

    def summary(self):
        self.model.summary()

    def train(self, dataset, query_imgs, epochs = 100):
        train_x, train_y = dataset
        q_x, q_y = query_imgs
 
        for i in range(epochs):
            q_idx = np.random.randint(0, len(q_y) - 1)
            query_img = q_x[q_idx]
            query_label = q_y[q_idx]
            
            query_img = np.expand_dims(query_img, axis=0)

            train_x = np.concatenate((train_x, query_img))
            # train_x = np.concatenate((train_x, query_img))

            train_x = np.expand_dims(train_x, axis = 3)

            train_x = np.expand_dims(train_x, axis = 0)

            print(train_x.shape)

            loss = self.model.train_on_batch(train_x, query_label)




rn_model = RelationNet(64, '')
# rn_model.summary()

ds = load_ds(64, 'train')

dataset, query_imgs = generate_data(ds, 2, 5)

rn_model.train(dataset, query_imgs)
