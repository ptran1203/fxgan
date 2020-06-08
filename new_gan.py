
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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

from keras.applications.vgg16 import VGG16

from keras.utils import np_utils
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

CATEGORIES_MAP = {
    'No Finding': 0,
    'Atelectasis': 1,
    'Effusion': 2,
    'Mass' :3,
    'Consolidation': 4,
    'Pneumothorax': 5,
    'Fibrosis': 6,
    'Infiltration': 7,
    'Emphysema': 8,
    'Nodule': 9,
    'Pleural_Thickening': 10,
    'Edema': 11,
    'Cardiomegaly': 12,
    'Hernia': 13,
    'Pneumonia': 14,
}

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
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
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
    imgs = []
    i = 0
    res = load_ds(resolution, 'train')
    if res:
        return res

    for file in os.listdir(DS_DIR + '/train/NORMAL'):
        path = DS_DIR + '/train/NORMAL/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs.append(get_img(path, resolution))
            labels.append(0)
        except:
            pass

    for file in os.listdir(DS_DIR + '/train/PNEUMONIA'):
        path = DS_DIR + '/train/PNEUMONIA/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs.append(get_img(path, resolution))
            labels.append(1)
        except:
            pass

    # channel last
    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (imgs.shape[0], resolution, resolution, 1)) # grayscale
    res = (imgs, np.array(labels))
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
    # channel last
    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (imgs.shape[0], resolution, resolution, 1)) # grayscale
    res = (imgs, np.array(labels))
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


## TRIPLET LOSS ##
def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    
    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance

## END TRIPLET LOSS ##

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
            # revert x = x * 127.5+127.5 / 255.0

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
            to_keep = [i for i, l in enumerate(y) if '|' not in l]
            to_keep = np.array(to_keep)
            x = x[to_keep]
            y = y[to_keep]
            x = np.expand_dims(x, axis=-1)
            to_train_classes = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule']
            if self.data_src == self.TEST:
                to_keep = np.array([i for i, l in enumerate(y) if l not in to_train_classes])
                x, y = x[to_keep], y[to_keep]
                self.dataset_x = x
                self.dataset_y = np.array([CATEGORIES_MAP[l] for l in y])
            else:
                to_keep = np.array([i for i, l in enumerate(y) if l in to_train_classes])
                x, y = x[to_keep], y[to_keep]
                self.dataset_x = x
                self.dataset_y = np.array([CATEGORIES_MAP[l] for l in y])

        # Normalize between -1 and 1
        self.dataset_x = (self.dataset_x - 127.5) / 127.5

        print(self.dataset_x.shape[0] , self.dataset_y.shape[0])
        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        # Prune
        if prune_classes:
            for class_to_prune in range(len(prune_classes)):
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
        classes = {k: [] for k in count.keys()}
        for c_id in count.keys():
            classes[c_id] = np.random.choice(self.per_class_ids[c_id], count[c_id])

        new_arr = []
        for i, label in enumerate(labels):
            idx, classes[label] = classes[label][-1], classes[label][:-1]
            new_arr.append(idx)

        return self.dataset_x[np.array(new_arr)]

    def other_labels(self, labels):
        clone = np.arange(labels.shape[0])
        clone[:] = labels
        for i in range(labels.shape[0]):
            to_get = self.classes[self.classes != labels[i]]
            clone[i] = to_get[np.random.randint(0, len(self.classes) - 1)]
        return clone

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
    D_RATE = 1
    def _res_block(self,
                  x,
                  units = 64,
                  kernel_size = 3,
                  activation = 'leaky_relu',
                  norm = 'batch',
                  scale=None,
                  bias=None):
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

        out = Conv2D(units, kernel_size, strides = 1, padding='same')(x)
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
                x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
                return x

    def _downscale(self, x, interpolation='conv', units=64,kernel_size=5):
        if interpolation == 'conv':
            # use convolution
            x = Conv2D(units, kernel_size, strides=2, padding='same')(x)
            return x
        else:
            # use upsamling layer
            x = MaxPooling2D()(x)
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

        code = AveragePooling2D()(x)
        code = Flatten()(code)

        self.attribute_encoder = Model(
            inputs = image,
            outputs = code
        )

    def build_features_from_classifier_model(self):
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        model_output = self.classifier.layers[-3](image)
        self.features_from_classifier = Model(
            inputs = image,
            output = model_output,
            name = 'Feature_matching_classifier'
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
        self.loss_type = loss_type
        if loss_type == 'binary':
            print('LOSS TYPE: BinaryCrossentropy')
            self.g_loss = keras.losses.BinaryCrossentropy()
            self.d_fake_loss = keras.losses.BinaryCrossentropy()
            self.d_real_loss = keras.losses.BinaryCrossentropy()
        elif loss_type == 'categorical':
            print('LOSS TYPE: sparse_categorical_crossentropy')
            self.g_loss = 'sparse_categorical_crossentropy'
            self.d_fake_loss = 'sparse_categorical_crossentropy'
            self.d_real_loss = 'sparse_categorical_crossentropy'
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
        self.build_latent_encoder()
        self.build_attribute_encoder()
        self.build_attribute_net()
        self.build_discriminator()
        self.build_features_from_d_model()
        self.build_res_unet()


        real_images = Input(shape=(self.resolution, self.resolution, self.channels))
        other_batch = Input(shape=(self.resolution, self.resolution, self.channels))
        positive_images = Input(shape=(self.resolution, self.resolution, self.channels))
        latent_code = Input(shape=(self.latent_size,))

        fake_images = Input(shape=(self.resolution, self.resolution, self.channels))

        real_output_for_d = self.discriminator([real_images, real_images])
        fake_output_for_d = self.discriminator([fake_images, other_batch])

        self.discriminator_model = Model(
            inputs = [real_images, other_batch, fake_images],
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
        self.latent_encoder.trainable = False
        self.attribute_encoder.trainable = True

        aux_fake = self.discriminator([fake, other_batch])

        self.combined = Model(
            inputs=[real_images, other_batch, latent_code],
            outputs=[aux_fake],
            name = 'Combined',
        )

        # fake_perceptual_features = self.vgg16_features(fake)
        # real_perceptual_features = self.vgg16_features(other_batch)

        # triplet function
        margin = 1.0
        anchor_code = self.latent_encoder(fake)
        pos_code = self.latent_encoder(other_batch)
        d_pos = K.mean(K.abs(anchor_code - pos_code))
        d_neg = K.mean(K.abs(anchor_code - self.latent_encoder(real_images)))

        self.combined.add_loss(2 * K.maximum(d_pos - d_neg + margin, 0.0))
        # self.combined.add_loss(K.mean(K.abs(anchor_code - pos_code)))

        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            metrics=['accuracy'],
            loss = self.g_loss,
        )


    def train_latent_encoder(self, bg_train, epochs = 100):
        save_path = '{}/latent_encoder.h5'.format(self.res_dir)
        if os.path.exists(save_path):
            print('Load latent_encoder')
            return self.latent_encoder.load_weights(save_path)
        print('Train latent_encoder')
        for e in range(epochs):
            losses = []
            for x, y, x2, y2 in bg_train.next_batch():
                flipped_y = bg_train.other_labels(y)
                pos_x = bg_train.get_samples_by_labels(y)
                neg_x = bg_train.get_samples_by_labels(flipped_y)
                out = self.latent_encoder.predict(x)
                loss = self.latent_encoder_trainer.train_on_batch([x, pos_x, neg_x], out)
                losses.append(loss)
            print('train attribute net epoch {} - loss: {}'.format(e, np.mean(np.array(losses))))

        self.latent_encoder.save(save_path)

    def _feature(self, x):
        return self.encoder(x)[-1]

    def vgg16_features(self, image):
        return self.perceptual_model(Concatenate()([
            image, image, image
        ]))

    def build_attribute_net(self):
        image = Input((self.resolution, self.resolution, self.channels))
        attr_feature = self.latent_encoder(image)

        scale = Dense(256, activation = 'relu')(attr_feature)
        scale = Dense(256, activation = 'relu')(scale)
        scale = Dense(1, name = 'norm_scale')(scale)

        bias = Dense(256, activation = 'relu')(attr_feature)
        bias = Dense(256, activation = 'relu')(bias)
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
            kernel_size = 3

            en_1 = Conv2D(64, kernel_size + 2, strides=2, padding="same")(image)
            en_1 = self._norm()(en_1)
            en_1 = actv(en_1)
            en_1 = Dropout(0.3)(en_1)

            en_2 = Conv2D(128, kernel_size, strides=2, padding="same")(en_1)
            en_2 = self._norm()(en_2)
            en_2 = actv(en_2)
            en_2 = Dropout(0.3)(en_2)

            en_3 = Conv2D(256, kernel_size, strides=2, padding='same')(en_2)
            en_3 = self._norm()(en_3)
            en_3 = actv(en_3)
            en_3 = Dropout(0.3)(en_3)

            en_4 = Conv2D(512, kernel_size, strides=2, padding='same')(en_3)
            en_4 = self._norm()(en_4)
            en_4 = actv(en_4)
            # en_4 = Dropout(0.3)(en_4)

            # content_code = self._res_block(en_4, 512, kernel_size, activation)
            # content_code = self._res_block(content_code, 512, kernel_size, activation)

            return Model(inputs = image, outputs = [en_2, en_3, en_4])

        image = Input(shape=(self.resolution, self.resolution, self.channels), name = 'image_1')
        image2 = Input(shape=(self.resolution, self.resolution, self.channels), name = 'image_2')

        latent_code = Input(shape=(128,), name = 'latent_code')

        self.encoder = _encoder()
        feature = self.encoder(image)        
        scale, bias = self.attribute_net(image2)

        decoder_activation = LeakyReLU()
        kernel_size = 3

        de = self._res_block(feature[2], 512, kernel_size,
                            norm='fn',
                            scale=scale, bias=bias)
        de = self._upscale(de, 'conv', 512, kernel_size)
        de = decoder_activation(de)
        de = Add()([de_1, feature[1]])

        de = self._res_block(de, 256, kernel_size,
                                norm='fn',
                                scale=scale, bias=bias)
        de = self._upscale(de, 'conv', 256, kernel_size)
        de = decoder_activation(de)
        de = Add()([de, feature[0]])

        de = self._res_block(de, 128, kernel_size,
                                norm='fn',
                                scale=scale, bias=bias)
        de = self._upscale(de, 'conv', 128, kernel_size)
        de = decoder_activation(de)

        de = self._res_block(de, 64, kernel_size,
                                norm='fn',
                                scale=scale, bias=bias)
        de = self._upscale(de, 'conv', 64, kernel_size)
        de = decoder_activation(de)

        final = Conv2D(1, kernel_size, strides=1, padding='same')(de)
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
            plt.plot(train_g, label='train_g_loss')
            plt.plot(test_g, label='test_g_loss')
            plt.ylabel('acc')
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
 
        plot_g(train_g, test_g)
        plot_d(train_d, test_d)


    def _build_common_encoder(self, image):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()
        kernel_size = 5

        cnn.add(Conv2D(64, kernel_size, strides=2, padding='same'))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))
        # 32 * 32 * 64

        cnn.add(keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))

        cnn.add(Conv2D(128, kernel_size, padding='same', strides=(2, 2)))
        self.loss_type == 'wasserstein_loss' and cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))
        # 16 * 16 * 128

        cnn.add(Conv2D(256, kernel_size, padding='same', strides=(2, 2)))
        # cnn.add(SelfAttention(256))
        self.loss_type == 'wasserstein_loss' and cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.3))
        # 8 * 8 * 256

        cnn.add(Conv2D(512, kernel_size, padding='same', strides=(2, 2)))
        self.loss_type == 'wasserstein_loss' and cnn.add(self._norm())
        cnn.add(LeakyReLU(alpha=0.2))

        # cnn.add(Flatten())

        features = cnn(image)
        return features

    def build_discriminator(self):
        resolution = self.resolution
        channels = self.channels

        image = Input(shape=(resolution, resolution, channels))
        other_batch = Input(shape=(resolution, resolution, channels))

        # scale bias for feature norm
        scale, bias = self.attribute_net(other_batch)

        features = self._build_common_encoder(image)
        features = FeatureNorm()([features, scale, bias])
        features = Dropout(0.3)(features)

        features = Flatten()(features)

        activation = 'sigmoid' if self.loss_type == 'binary' else 'linear'
        if self.loss_type == 'categorical':
            aux = Dense(self.nclasses + 1, activation = 'softmax', name='auxiliary')(features)
        else:
            aux = Dense(
                1, activation = activation,name='auxiliary'
            )(features)

        self.discriminator = Model(inputs=[image, other_batch],
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
            flipped_labels = bg_train.other_labels(label_batch)
            other_batch = bg_train.get_samples_by_labels(flipped_labels)
            for i in range(self.D_RATE):
                generated_images = self.generator.predict(
                    [
                        image_batch,
                        other_batch,
                        f,
                    ],
                    verbose=0
                )

                # X, aux_y = self.shuffle_data(X, aux_y)
                fake_label = np.ones((generated_images.shape[0], 1))
                real_label = -np.ones((label_batch.shape[0], 1))
                real_label_for_d = -np.ones((label_batch.shape[0], 1))

                if self.loss_type == 'binary':
                    real_label *= 0
                    real_label_for_d *= 0
                if self.loss_type == 'categorical':
                    real_label = flipped_labels
                    real_label_for_d = label_batch
                    fake_label = np.full(label_batch.shape[0], self.nclasses)

                loss, acc, *rest = self.discriminator_model.train_on_batch(
                    [image_batch, other_batch, generated_images],
                    [fake_label, real_label_for_d]
                )
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################
            f = self.generate_latent(range(crt_batch_size))
            # positive_images = bg_train.get_samples_by_labels(flipped_labels)
            [loss, acc, *rest] = self.combined.train_on_batch(
                [image_batch, other_batch, f],
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
            # self.train_latent_encoder(bg_train)
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
                rand_x, rand_y = self.shuffle_data(bg_test.dataset_x, bg_test.dataset_y)

                generated_images = self.generator.predict(
                    [
                        bg_test.dataset_x,
                        rand_x,
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

                if self.loss_type == 'binary':
                    real_label *= 0
                if self.loss_type == 'categorical':
                    real_label = rand_y
                    fake_label = np.full(generated_images.shape[0], self.nclasses)

                X = [bg_test.dataset_x,rand_x, generated_images]
                Y = [fake_label, real_label]

                test_disc_loss, test_disc_acc, *rest = self.discriminator_model.evaluate(X, Y, verbose=False)

                [test_gen_loss, test_gen_acc, *rest] = self.combined.evaluate(
                    [
                        bg_test.dataset_x,
                        rand_x,
                        f
                    ],
                    [real_label],
                    verbose = 0
                )

                if e % 25 == 0:
                    self.evaluate_d(np.concatenate([X[0], X[2]], axis=0), np.concatenate(Y, axis=0))
                    self.evaluate_g(
                        [
                            bg_test.dataset_x,
                            rand_x,
                            f,
                            
                        ],
                        [real_label],
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

                    # calculate attribute distance
                    self.plot_loss_his()
                    self.plot_acc_his()
                    self.plot_feature_distr(bg_train)

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

    def plot_feature_distr(self, bg):
        pca = PCA(n_components=2)
        x, y = bg.dataset_x, bg.dataset_y

        class_1 = bg.get_samples_for_class(0, 100)
        class_2 = bg.get_samples_for_class(1, 100)
        fake_1 = self.generator.predict([class_1,
                                       class_2,
                                       self.generate_latent(range(100))])

        fake_2 = self.generator.predict([class_2,
                                       class_1,
                                       self.generate_latent(range(100))])

        def _plot_pca(x, y, encoder, name):
            step = 1
            x_embeddings = encoder.predict(x)
            decomposed_embeddings = pca.fit_transform(x_embeddings)
            fig = plt.figure(figsize=(16, 8))
            for label in np.unique(y):
                decomposed_embeddings_class = decomposed_embeddings[y == label]
                plt.subplot(1,2,2)
                plt.scatter(decomposed_embeddings_class[::step, 1],
                            decomposed_embeddings_class[::step, 0],
                            label=str(label))
                plt.title(name)
                plt.legend()
            plt.show()

        # latent_encoder
        imgs = np.concatenate([x, fake_1, fake_2])
        show_samples(np.concatenate([fake_1[0:2], fake_2[0:2]]))
        labels = np.concatenate([y, np.full((100,), 'fake of 1'),  np.full((100,), 'fake of 0')])
    
        _plot_pca(imgs, labels, self.latent_encoder, 'latent encoder')
        # attribute encoder
        _plot_pca(imgs, labels, self.attribute_encoder, 'attribute   encoder')



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