
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
    Average,
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
        query_size = 95,
        c_way = 2,
        k_shot = 5,
    ):
        self.batch_size = batch_size
        self.data_src = data_src
        self.c_way = c_way
        self.k_shot = k_shot
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

        if self.data_src == self.TRAIN:
            self.build_dataset(query_size)

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

    def next_query_batch(self):
        """
        Next support and query batch
        call:
            for query_x, query_y in BatchGen.next_query_batch():
                # do business
        """
        indices = np.arange(self.query_x.shape[0])
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.query_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]

            yield  self.query_x[access_pattern],  self.query_y[access_pattern]

    def build_dataset(self, query_size):
        idxs = []
        qidxs = []
        train_x, train_y = self.dataset_x, self.dataset_y
        for i in range(self.c_way):
            idx = np.where(train_y == i)[0]
            np.random.shuffle(idx)
            qidx = idx[self.k_shot: query_size + self.k_shot]
            
            if i == self.c_way - 1:
                idxs.append(idx[:self.k_shot + 1])
            else:
                idxs.append(idx[:self.k_shot])
            
            qidxs.append(qidx)

        s_idx = np.concatenate(idxs)
        q_idx = np.concatenate(qidxs)

        np.random.shuffle(q_idx)

        self.query_x = train_x[q_idx]
        self.query_y = train_y[q_idx]

        self.support_x = train_x[s_idx]
        self.support_y = train_y[s_idx]

        save_image_array(self.support_x, None, True)
        print(self.support_y)

        print('Query size: ', self.query_x.shape[0])
        print('Support size: ', self.support_x.shape[0])

    def merge_support_images(self, support_fakes ,repeats = None):
        if repeats is None:
            repeats = self.batch_size

        imgs = np.concatenate((self.support_x, support_fakes, np.zeros(
            (1,64,64,1)
        )))
        imgs = np.expand_dims(imgs, axis = 0)
        return np.repeat(
                    imgs, repeats, axis= 0
                )

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
    
    def plot_classifier_acc(self):
        plt.plot(self.classifier_acc, label='classifier_acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch(x5)')
        plt.legend()
        plt.show()

    def build_generator(self, latent_size, init_resolution=8):
        resolution = self.resolution
        channels = self.channels
        init_channels  = 128
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

    def _embedding_module(self):
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        model.add(Dropout(0.3))
    
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.name = 'embedding_module'
        return model
    
    def _relation_module(self):
        model = Sequential()

        model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same'))

        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Flatten())
        model.add(Dropout(0.4))

        # model.add(Dense(8, activation='relu' ))
        model.add(Dense(1, activation='sigmoid'))
        model.name = 'relation_module'
        return model

    def build_reconstructor(self, latent_size):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(resolution, resolution, channels))
        features = self._embedding_module()(image)
        f_features = Flatten()(features)
        # Reconstructor specific
        latent = Dense(latent_size, activation='linear')(f_features)
        self.reconstructor = Model(inputs=image, outputs=latent, name = 'reconstructor')

    def build_discriminator(self):
        resolution = self.resolution
        channels = self.channels
        support_images = Input(shape = (
            self.c_way * self.k_shot + 1,
            self.resolution,
            self.resolution,
            self.channels,
        ))

        images = Input(shape = (resolution, resolution, channels))

        embedding_module = self._embedding_module()
        relation_module = self._relation_module()

        features = embedding_module(images)
        support_features = [[] for i in range(self.c_way)]

        idx = 0
        for classid in range(self.c_way):
            for _ in range(self.k_shot):
                support_features[classid].append(
                    embedding_module(Lambda(lambda x: x[:,idx,:,:,:])(support_images))
                )
                idx += 1

        relation_scores = []

        for classid in range(self.c_way):
            sfeatures = Average()(support_features[classid])
            concat_features = Concatenate()([sfeatures, features])
            relation_scores.append(relation_module(concat_features))

        outputs = Concatenate()(relation_scores)

        self.discriminator = Model(
            inputs = [support_images, images],
            outputs = outputs,
            name = 'Discriminator'
        )

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

    def discriminate(self, support_images, images):
        return self.discriminator([support_images, images])

    def __init__(self, classes, target_class_id,
                # Set dratio_mode, and gratio_mode to 'rebalance' to bias the sampling toward the minority class
                # No relevant difference noted
                dratio_mode="uniform", gratio_mode="uniform",
                adam_lr=0.00005, latent_size=100,
                res_dir = "./res-tmp", image_shape=[3,32,32], min_latent_res=8,
                c_way = 2, k_shot = 5):
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.c_way = c_way
        self.k_shot = k_shot
        self.classes = classes
        self.target_class_id = target_class_id  # target_class_id is used only during saving, not to overwrite other class results.
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[-1]
        self.resolution = image_shape[0]
        self.min_latent_res = min_latent_res

        self.min_latent_res = min_latent_res
        self.classifier = load_classifier(self.resolution)
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy'])
        self.classifier_acc = pickle_load(CLASSIFIER_DIR + '/acc_array.pkl') or []

        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build generator
        self.build_generator(latent_size)
        self.generator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

        latent_gen = Input(shape=(latent_size, ))
        support_images = Input(shape = (
            self.c_way * self.k_shot + 1,
            self.resolution,
            self.resolution,
            self.channels,
        ))

        # Build discriminator
        self.build_discriminator()
        self.discriminator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics=['accuracy'],
            loss='mean_squared_error'
        )

        # Build reconstructor
        self.build_reconstructor(latent_size)
        self.reconstructor.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

        # Define combined for training generator.
        fake = self.generator(latent_gen)

        self.discriminator.trainable = False
        self.reconstructor.trainable = False
        self.generator.trainable = True
        aux = self.discriminate(support_images ,fake)

        self.combined = Model(
            inputs=[latent_gen, support_images],
            outputs=aux,
            name = 'Combined'
        )

        self.combined.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            metrics=['accuracy'],
            loss='mse'
        )

        # Define initializer for autoencoder
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.reconstructor.trainable = True

        img_for_reconstructor = Input(shape=(self.resolution, self.resolution,self.channels))
        img_reconstruct = self.generator(self.reconstructor(img_for_reconstructor))
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

        for image_batch, label_batch in bg_train.next_query_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################

            fake_size = int(np.ceil(crt_batch_size * 1.0/self.nclasses)) + self.k_shot
    
            # sample some labels from p_c, then latent and images
            sampled_labels = self._biased_sample_labels(fake_size, "d")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            generated_images = self.generator.predict(latent_gen, verbose=0)
            fake_images = generated_images[self.k_shot:]
            support_fakes = generated_images[:self.k_shot]

            X = np.concatenate((image_batch, fake_images))
            support_images = bg_train.merge_support_images(self.support_fakes, X.shape[0])
            aux_y = np.concatenate((label_batch, np.full(len(fake_images) , self.nclasses )), axis=0)
            aux_y = np_utils.to_categorical(aux_y, self.nclasses + 1)
            
            X, aux_y = self.shuffle_data(X, aux_y)
            loss, acc = self.discriminator.train_on_batch([support_images, X], aux_y)
    
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################
            sampled_labels = self._biased_sample_labels(fake_size - self.k_shot + crt_batch_size, "g")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            sampled_labels = np_utils.to_categorical(sampled_labels, self.nclasses + 1)
            latent_gen, sampled_labels = self.shuffle_data(latent_gen, sampled_labels)
            loss, acc = self.combined.train_on_batch([latent_gen, support_images], sampled_labels)
            epoch_gen_loss.append(loss)
            epoch_gen_acc.append(acc)

        # return statistics: generator loss,
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
                for image_batch, label_batch in bg_train.next_query_batch():

                    autoenc_train_loss_crt.append(self.autoenc_0.train_on_batch(image_batch, image_batch))
                autoenc_train_loss.append(np.mean(np.array(autoenc_train_loss_crt), axis=0))

            autoenc_loss_fname = "{}/{}_autoencoder.csv".format(self.res_dir, self.target_class_id)
            with open(autoenc_loss_fname, 'w') as csvfile:
                for item in autoenc_train_loss:
                    csvfile.write("%s\n" % item)

            self.generator.save(generator_fname)
            self.reconstructor.save(reconstructor_fname)

        layers_embedding = self.reconstructor.get_layer('embedding_module').layers
        layers_d = self.discriminator.get_layer('embedding_module').layers

        for l in range(1, len(layers_embedding)-1):
            layers_d[l].set_weights( layers_embedding[l].get_weights())

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
        self.support_fakes = self.generator.predict(
                    self.generate_latent([0,0,0,1,1]), verbose=False)
        print('Init support fakes ', self.support_fakes.shape[0])
        save_image_array(self.support_fakes, None, True)

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
        pickle_save(self.classifier_acc, CLASSIFIER_DIR + '/acc_array.pkl')

    def evaluate_d(self, support_x, test_x, test_y):
        y_pre = self.discriminator.predict([support_x, test_x])
        y_pre = np.argmax(y_pre, axis=1)
        test_y = np.argmax(test_y, axis=1)
        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)  # shape=(12, 12)
        plt.figure()
        plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
        plt.show()

    def evaluate_g(self, support_x, test_x, test_y):
        y_pre = self.combined.predict([test_x, support_x])
        y_pre = np.argmax(y_pre, axis=1)
        test_y = np.argmax(test_y, axis=1)
        cm = metrics.confusion_matrix(y_true=test_y, y_pred=y_pre)  # shape=(12, 12)
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
                        self.reconstructor.predict(
                            act_img_samples
                        )
                    ),
                    self.generate_samples(crt_c, 10, bg_train)
                ]
            ])
            for crt_c in range(1, self.nclasses):
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

            save_image_array(img_samples, None, True)

            # Train
            for e in range(start_e, epochs):
                start_time = datetime.datetime.now()
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc = self._train_one_epoch(bg_train)

                # Test: # generate a new batch of noise
                nb_test = bg_test.get_num_samples()
                fake_size = int(np.ceil(nb_test * 1.0/self.nclasses))
                sampled_labels = self._biased_sample_labels(nb_test + self.k_shot, "d")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
            
                # sample some labels from p_c and generate images from them
                generated_images = self.generator.predict(
                    latent_gen, verbose=False)

                fake_images = generated_images[self.k_shot:]
                support_fakes = generated_images[:self.k_shot]

                generated_images[0] = (generated_images[0] * 127.5 + 127.5).astype(np.uint8)
                cv2_imshow(generated_images[0])
            
                X = np.concatenate( (bg_test.dataset_x, fake_images) )
                aux_y = np.concatenate((bg_test.dataset_y, np.full(len(fake_images), self.nclasses )), axis=0)
                aux_y = np_utils.to_categorical(aux_y, self.nclasses + 1)
                support_images = bg_train.merge_support_images(self.support_fakes, X.shape[0])
                # see if the discriminator can figure itself out...
                test_disc_loss, test_disc_acc = self.discriminator.evaluate(
                    [support_images, X], aux_y, verbose=False)

            
                # make new latent
                sampled_labels = self._biased_sample_labels(fake_size + nb_test, "g")
                latent_gen = self.generate_latent(sampled_labels, bg_test)

                support_images_g = bg_train.merge_support_images(self.support_fakes,
                                                               sampled_labels.shape[0])


                test_y = np_utils.to_categorical(
                        sampled_labels,
                        self.nclasses + 1
                    )
                test_gen_loss, test_gen_acc = self.combined.evaluate(
                    [latent_gen, support_images_g],
                    test_y, verbose=False)

                if e % 5 == 0:
                    print('Evaluate D')
                    self.evaluate_d(support_images, X, aux_y)
                    print('Evaluate G')
                    self.evaluate_g(support_images_g, latent_gen, test_y)


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

                # Save sample images
                if e % 15 == 0:
                    img_samples = np.array([
                        self.generate_samples(c, 10, bg_train)
                        for c in range(0,self.nclasses)
                    ])

                    save_image_array(
                        img_samples,
                        '{}/plot_class_{}_epoch_{}.png'.format(self.res_dir, self.target_class_id, e),
                        show=True
                    )

                # Generate whole evaluation plot (real img, autoencoded img, fake img)
                if e % 10 == 5:
                    self.plot_loss_his()
                    self.plot_acc_his()
                    self.backup_point(e)
                    crt_c = 0
                    sample_size = 10
                    img_samples = self.generate_samples(crt_c, sample_size, bg_train)
                    five_imgs = img_samples[:5]
                    for crt_c in range(1, self.nclasses):
                        new_samples = self.generate_samples(crt_c, sample_size, bg_train)
                        five_imgs = np.concatenate((five_imgs, new_samples[:5]), axis=0)
                    
                    save_image_array(five_imgs, None, True)
            self.trained = True

    def generate_samples(self, c, samples, bg = None):
        return self.generate(np.full(samples, c), bg)

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


