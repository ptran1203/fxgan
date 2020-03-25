

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
    BatchNormalization,Activation
)
from keras.layers import multiply as kmultiply
from keras.layers import add as kadd
import csv
from PIL import Image


# Hyper Parameters
FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 2
SAMPLE_NUM_PER_CLASS = 100
BATCH_NUM_PER_CLASS = 10
EPISODE = 5000
TEST_EPISODE = 600
LEARNING_RATE = 0.001
HIDDEN_UNIT = 10


def embedding_module(img_size, hidden_size):
    # input
    input_tensor = Input((img_size, img_size, 1))

    x = Conv2D(64, (3, 3))(input_tensor)
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

    features = Flatten()(x)
    return Dense(hidden_size)(features)

def relation_module(input_size, hidden_size):
    # input is vector
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_size,)))
    model.add(Conv2D(64, (3, 3), strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    # relation score
    model.add(Activation('sigmoid'))
    return model




