
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
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Dropout,
    BatchNormalization, Activation,
    Lambda, Layer, Add, Concatenate,
    Average, GaussianNoise,
    AveragePooling2D, GlobalAveragePooling2D,
)
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.applications.vgg16 import VGG16
import sklearn.metrics as metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import datetime
import cv2
import utils
import logger
from const import BASE_DIR

K.common.set_image_dim_ordering('tf')

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))

def l2_distance(a, b):
    return np.mean(np.square(a - b))

def cosine_sim(a, b):
    return - (np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


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

        if 'bn' in self.norm:
            logger.info('Use Batch norm for FeatureNorm layer')
            axis = [0, 1, 2]
        else:
            # instance norm
            logger.info('Use Instance norm for FeatureNorm layer')
            axis = [1, 2]

        mean = K.mean(x, axis = axis, keepdims = True)
        std = K.std(x, axis = axis, keepdims = True)
        norm = (x - mean) * (1 / (std + self.epsilon))

        broadcast_scale = K.reshape(scale, (-1, 1, 1, C))
        broadcast_bias = K.reshape(bias, (-1, 1, 1, C))

        return norm * broadcast_scale + broadcast_bias

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

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

class FXGAN:
    D_RATE = 1
    def _triple_tensor(self, x):
        """
        Convert 1 channel image to 3 channels image
        """
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


    def _dc_block(self, x, units, 
                kernel_size=3,activation='relu',
                norm='batch'):
        x = Conv2DTranspose(units, kernel_size, strides=2, padding='same')(x)
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
                                                    [classid] * samples)
        else:
            print("same images")
            support_images = bg.ramdom_kshot_images(self.k_shot,
                                                    [classid])
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
        print("predicts: ", self.classify_by_metric(bg, generated_images))
        utils.show_samples(support_images)
        utils.show_samples(generated_images)

    def build_latent_encoder(self):
        fname = '{}/{}/latent_encoder_{}'.format(BASE_DIR,
                                                self.dataset,
                                                self.resolution)
        if self.env != "colab":
            fname = "/content/latent_encoder_128"

        json_file = open(fname + '.json', 'r')
        model = json_file.read()
        json_file.close()
        self.latent_encoder = model_from_json(model)
        modified = os.path.getmtime(fname + '.json')
        print('Latent model modified at: ',
            datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S'))
        self.latent_encoder.load_weights(fname + '.h5')
        self.latent_encoder.trainable = False


    def classify_by_metric(self, bg, images, metric='l2', bg_test=None, anchor=None):
        # currently do one-shot classification
        size = len(bg.classes)
        if bg_test is not None:
            size += len(bg_test.classes)

        sp_vectors = self.means[:size].reshape(-1, 1, self.latent_size)
        if anchor is not None:
            anchors = self.latent_code(utils.triple_channels(anchor[0]))
        vectors = self.latent_code(utils.triple_channels(images))
        metric_func = l2_distance if metric == 'l2' else cosine_sim
        similiarity = []
        for i, vector in enumerate(vectors):
            for c, sp_vector in enumerate(sp_vectors):
                if anchor is not None and c == anchor[1]:
                    similiarity.append(metric_func(vector, anchors[i]))
                else:
                    similiarity.append(metric_func(vector, sp_vector))
    
        pred = np.argmin(np.array(similiarity).reshape(-1, size), axis=1)
        return pred


    def gen_for_class(self, bg, bg_test=None, classid=0,size=1000):
        total = None
        for i in range(1000):
            labels = [classid] * size
            labels = np.array(labels)
            latent = self.generate_latent(labels)
            if classid in bg.classes:
                support = bg.ramdom_kshot_images(self.k_shot,
                                                np.full(size, classid))
            else:
                if bg_test is None:
                    raise("bg_test is None, please give it, boi")

                support = bg_test.ramdom_kshot_images(self.k_shot,
                                                np.full(size, classid))

            gen = self.generate(support, latent)
            d_outputs = self.classify_by_metric(bg, gen, bg_test=bg_test, anchor=(support, classid))
            to_keep = np.where(labels == d_outputs)[0]
            gen = gen[to_keep]
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

    def evaluate_by_metric(self, bg, images, labels, metric='l2'):
        pred = self.classify_by_metric(bg, images, metric)
        acc = (pred == labels).mean()
        return acc


    def compute_multivariate(self, bg):
        print("Computing feature distribution")
        if not hasattr(self, 'covariances'):
            self.covariances = []
            self.means = []
        else:
            self.covariances = list(self.covariances)
            self.means = list(self.means)

        for c in np.unique(bg.dataset_y):
            imgs = bg.dataset_x[bg.per_class_ids[c]]
            imgs = utils.triple_channels(imgs)
            latent = self.latent_code(imgs)
            
            self.covariances.append(np.cov(np.transpose(latent)))
            self.means.append(np.mean(latent, axis=0))

        self.covariances = np.array(self.covariances)
        self.means = np.array(self.means)


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
                k_shot=5, sampling='normal',
                advance_losses={'triplet': 0.1},
                env="colab",
                ):
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
        self.env=env
        # normal: sampling from normal distribution
        # code: sampling from latent code distribution (computed by classifier)
        self.sampling = sampling
        self.advance_losses = advance_losses

        self.norm = norm
        self.loss_type = loss_type

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
        self.build_discriminator()
        self.build_features_from_d_model()
        if self.resnet:
            self.build_resnet_generator()
        else:
            self.build_dc_generator()

        if self.loss_type == 'categorical':
            self.discriminator.compile(
                optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
                metrics = ['accuracy'],
                loss = 'sparse_categorical_crossentropy'
            )

        real_images = Input(shape=(self.resolution, self.resolution, self.channels))
        # Use VGG16 model -> channels = 3
        latent_code = Input(shape=(self.latent_size,))
        fake_images = Input(shape=(self.resolution, self.resolution, self.channels))

        if self.loss_type != 'categorical':
            real_output_for_d = self.discriminator(real_images)
            fake_output_for_d = self.discriminator(fake_images)

            self.discriminator_fake = Model(
                inputs = [fake_images],
                outputs = fake_output_for_d,
                name='D_fake',
            )
            self.discriminator_fake.compile(
                optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
                metrics = ['accuracy'],
                loss = self.d_fake_loss
            )

            self.discriminator_real = Model(
                inputs = [real_images],
                outputs = real_output_for_d,
                name='D_real',
            )
            self.discriminator_real.compile(
                optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
                metrics = ['accuracy'],
                loss = self.d_real_loss
            )

        # Define combined for training generator.
        real_images_for_G = Input((self.resolution, self.resolution, self.channels))
         # real attr
        attr_features = self.latent_encoder(self._triple_tensor(real_images_for_G))
        fake = self.generator([
            real_images_for_G, attr_features, latent_code
        ])

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.features_from_d_model.trainable = False

        aux_fake = self.discriminator(fake)

        negative_samples = Input((self.resolution,self.resolution,self.channels))
        fake_attribute = self.latent_encoder(self._triple_tensor(fake))

        self.combined = Model(
            inputs=[real_images_for_G, negative_samples, latent_code],
            outputs=aux_fake,
            name = 'Combined',
        )

        # triplet function
        margin = 1.0
        if 'triplet' in advance_losses:
            k_op = K.sum
        else:
            k_op =  K.mean

        d_pos = k_op(K.square(fake_attribute - attr_features), axis=1)
        d_neg = k_op(K.square(
                fake_attribute -
                self.latent_encoder(self._triple_tensor(negative_samples))
                ), axis=1)

        triplet = K.maximum(d_pos - d_neg + margin, 0.0)

        if 'triplet_D' in advance_losses:
            k_op_d = K.sum
        else:
            k_op_d =  K.mean

        # Recontruction loss
        recontruction_loss = K.square(fake - real_images_for_G)

        if 'triplet' in advance_losses:
            self.combined.add_loss(advance_losses['triplet'] * triplet)
        if 'l2_feat' in advance_losses:
            self.combined.add_loss(advance_losses['l2_feat'] * d_pos)
        if 'recon' in advance_losses:
            self.combined.add_loss(advance_losses['recon'] * K.mean(recontruction_loss))

        self.combined.compile(
            optimizer=Adam(
                lr=self.g_lr,
                beta_1=self.adam_beta_1
            ),
            metrics=['accuracy'],
            loss = self.g_loss,
        )
        self._show_settings()

    def encode_image(self, image):
        """
        Mapping image to latent code
        """
        kernel_size = 3

        x = Conv2D(32, kernel_size, strides = 2, padding='same')(image)
        x = self._norm()(x)
        x = Activation('relu')(x)

        # until downsample to 4x4
        connections = [x]
        channels = 64
        width = x.shape[-2]
        i = 0
        while width != 4:
            x_temp = Conv2D(channels, kernel_size, strides=2, padding='same')(connections[i])
            x_temp = self._norm()(x_temp)
            x_temp = Activation('relu')(x_temp)
            connections.append(x_temp)
            width //= 2
            channels *= 2
            i += 1

        code = GlobalAveragePooling2D()(connections[-1])

        return connections, code

    def build_dc_generator(self):
        init_channels = self.resolution * 4
        latent_code = Input(shape=(self.latent_size,), name = 'latent_code')
        attribute_code = Input(shape=(self.latent_size,), name = 'attribute_code')
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        activation = 'relu'
        # connections, content_code = self.encode_image(image)

        latent_vector = Concatenate()([attribute_code, latent_code])
        latent_vector = GaussianNoise(0.1)(latent_vector)

        latent = Dense(4 * 4 * init_channels)(latent_vector)
        latent = self._norm()(latent)
        latent = Activation(activation)(latent)
        latent = Reshape((4, 4, init_channels))(latent)

        kernel_size = 5
        de = self._dc_block(latent, init_channels, kernel_size,
                            activation=activation,
                            norm='in')
        de = self._dc_block(de, init_channels, kernel_size,
                            activation=activation,
                            norm='in')

        if self.attention:
            de = SelfAttention(init_channels)(de)

        output_shape = 16
        while output_shape != self.resolution // 2:
            init_channels //= 2
            output_shape *= 2
            de = self._dc_block(de, init_channels, kernel_size,
                            activation=activation,
                            norm='in')

        final = self._dc_block(de, self.channels, kernel_size,
                        activation='tanh',
                        norm=None)

        self.generator = Model(
            inputs = [image, attribute_code, latent_code],
            outputs = final,
            name='dc_gen'
        )


    def build_resnet_generator(self):
        init_channels = self.resolution * 4
        latent_code = Input(shape=(self.latent_size,), name = 'latent_code')
        attribute_code = Input(shape=(self.latent_size,), name = 'attribute_code')
        image = Input(shape=(self.resolution, self.resolution, self.channels))
        activation = 'relu'
        # connections, content_code = self.encode_image(image)

        latent_vector = Concatenate()([attribute_code, latent_code])
        latent_vector = GaussianNoise(0.1)(latent_vector)

        latent = Dense(4 * 4 * init_channels)(latent_vector)
        latent = self._norm()(latent)
        latent = Activation(activation)(latent)
        latent = Reshape((4, 4, init_channels))(latent)

        kernel_size = 3
        interpolation = 'nearest'
        # de = Add()([de, connections[-1]])
        de = self._up_resblock(latent, init_channels, kernel_size,
                            activation=activation,
                            norm='in')
        # de = Add()([de, connections[-2]])
        de = self._up_resblock(de, init_channels, kernel_size,
                            activation=activation,
                            norm='in')
        # de = Add()([de, connections[-3]])

        if self.attention:
            de = SelfAttention(init_channels)(de)

        while de.shape[-2] != self.resolution:
            init_channels //= 2
            de = self._up_resblock(de, init_channels, kernel_size,
                                activation=activation,
                                norm='in')

        de = self._norm()(de)
        de = Activation('relu')(de)

        final = Conv2D(self.channels, kernel_size, strides=1, padding='same')(de)
        outputs = Activation('tanh')(final)

        self.generator = Model(
            inputs = [image, attribute_code, latent_code],
            outputs = outputs,
            name='resnet_gen'
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
        def plot_g(train_g, test_g):
            plt.plot(train_g, label='train adv')
            plt.plot(test_g, label='test adv')
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
        logger.info('- Dataset: {}'.format(self.dataset))
        logger.info('- Num of classes: {}'.format(self.nclasses))
        logger.info('- Generator: {}'.format(self.generator.name))
        logger.info('- Self-Attention: {}'.format(self.attention))
        logger.info('- K-shot: {}'.format(self.k_shot))
        logger.info('- Adverasial loss: {}'.format(self.loss_type))
        if 'batch' in self.norm:
            norm_type = 'Batch norm'
        else:
            norm_type = 'Instance norm'
        logger.info('- Normalization: {}'.format(norm_type))
        fn_norm = 'fn' in self.norm
        logger.info('- Use feature normaliztion: {}'.format(fn_norm))
        print('- Advance losses: ', self.advance_losses)
        print('\n==================================================\n')


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
    
        if self.attention:
            x = SelfAttention(128)(x)

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
        # attr_image = Input(shape=(self.k_shot, resolution, resolution, 3))

        features = self._discriminator_feature(image)
        features = Dropout(0.4)(features)
        activation = 'sigmoid' if self.loss_type == 'binary' else 'linear'
        last_channels = 1
        if self.loss_type == 'categorical':
            activation = 'softmax'
            last_channels = self.nclasses + 1

        aux = Dense(
            last_channels, activation = activation,name='auxiliary'
        )(features)

        self.discriminator = Model(inputs=image,
                                   outputs=aux,
                                   name='discriminator')


    def generate_latent(self, c, size = 1):
        if self.sampling == 'code':
            return np.array([
                np.random.multivariate_normal(self.means[e], self.covariances[e])
                for e in c
            ])

        return np.array([
            np.random.normal(0, 1, self.latent_size)
            for i in c
        ])


    def build_features_from_d_model(self):
        self.features_from_d_model = Model(
            inputs = self.discriminator.inputs,
            output = self.discriminator.layers[-2].get_output_at(-1),
            name = 'Feature_matching'
        )

    def _norm(self):
        return BatchNormalization() if 'batch' in self.norm else InstanceNormalization()

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        class_weight = None #bg_train.class_weights
        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################
            fake_size = crt_batch_size // self.nclasses
            f = self.generate_latent(label_batch)
            k_shot_batch = bg_train.ramdom_kshot_images(self.k_shot, label_batch)
            for i in range(self.D_RATE):
                generated_images = self.generate(k_shot_batch, f)

                fake_label = np.ones((crt_batch_size, 1))
                real_label = -np.ones((crt_batch_size, 1))

                if self.loss_type == 'binary':
                    real_label *= 0
                if self.loss_type == 'categorical':
                    real_label = label_batch
                    loss, acc = self.discriminator.train_on_batch(
                        np.concatenate([image_batch, generated_images], axis=0),
                        np.concatenate([
                            real_label,
                            np.full(crt_batch_size, self.nclasses)], axis=0),
                        class_weight=class_weight,
                    )
                else:
                    loss_fake, acc_fake, *rest = \
                            self.discriminator_fake.train_on_batch([generated_images],
                                                                    fake_label)
                    loss_real, acc_real, *rest = \
                            self.discriminator_real.train_on_batch([image_batch],
                                                                    real_label)
                    loss = 0.5 * (loss_fake + loss_real)
                    acc = 0.5 * (acc_fake + acc_real)

            epoch_disc_loss.append(loss)

            ################## Train Generator ##################
            f = self.generate_latent(label_batch)
            negative_samples = bg_train.get_samples_by_labels(bg_train.other_labels(label_batch))
            gloss, gacc = self.combined.train_on_batch(
                [
                    image_batch,
                    negative_samples, f
                ],
                real_label,
                class_weight=class_weight if self.loss_type == 'categorical' else None
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

    def plot_cm_for_G(self, bg, bg_test=None, labels=None, metric='l2'):
        if labels is None:
            labels = bg.dataset_y
            if bg_test is not None:
                labels = np.concatenate([labels, bg_test.dataset_y])
        else:
            labels = np.array(labels)

        train_max_id = np.max(bg.classes)
        train_mask = np.where(labels <= train_max_id)[0]
        test_mask = np.where(labels > train_max_id)[0]

        support_images = bg.ramdom_kshot_images(self.k_shot,
                                labels[train_mask])
        if bg_test is not None:
            support_images = np.concatenate([support_images,
                bg_test.ramdom_kshot_images(self.k_shot,
                                labels[test_mask])])


        latent = self.generate_latent(labels)
        generated_images = self.generate(support_images, latent)
        pred = self.classify_by_metric(bg, generated_images, metric, bg_test)
        cm = metrics.confusion_matrix(y_true=labels, y_pred=pred)
        plt.figure()
        plot_confusion_matrix(cm,
                              hide_ticks=False,
                              cmap=plt.cm.Blues,
                              figsize=(8,8))
        plt.show()


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
        attr_code = self.latent_code(image)
        return self.generator.predict([
            image[:,:,:,:self.channels], attr_code, latent
        ])


    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            # Initialization
            print("init gan")
            self.compute_multivariate(bg_train)
            self.compute_multivariate(bg_test)
            start_e = self.init_gan()
            # self.init_autoenc(bg_train)
            print("gan initialized, start_e: ", start_e)

            crt_c = 0
            # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                        np.full(10, crt_c))
            f = self.generate_latent([crt_c] * 10)
            img_samples = np.array([
                [
                    act_img_samples[:,:,:,:self.channels],
                    self.generate(act_img_samples, f)
                ]
            ])
            for crt_c in range(1, min(self.nclasses, 3)): # more 3 classes
                # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                            np.full(10, crt_c))
                new_samples = np.array([
                    [
                        act_img_samples[:,:,:,:self.channels],
                        self.generate(act_img_samples, f)
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
                test_size = 100 * self.nclasses
                random_ids = np.arange(bg_train.dataset_y.shape[0])
                np.random.shuffle(random_ids)
                random_ids = random_ids[:test_size]
                test_batch_x = bg_train.dataset_x[random_ids]
                test_batch_y = bg_train.dataset_y[random_ids]
                k_shot_test_batch = bg_train.ramdom_kshot_images(self.k_shot, test_batch_y)
                f = self.generate_latent(test_batch_y)

                generated_images = self.generate(k_shot_test_batch, f)

                X = np.concatenate([test_batch_x, generated_images])

                fake_label = np.ones((generated_images.shape[0], 1))
                real_label = -np.ones((test_size, 1))

                if self.loss_type == 'binary':
                    real_label *= 0
                if self.loss_type == 'categorical':
                    real_label = test_batch_y
                    fake_label = np.full(generated_images.shape[0], self.nclasses)

                X = [test_batch_x, generated_images]
                Y = [fake_label, real_label]

                if self.loss_type == 'categorical':
                    test_disc_loss, _ = self.discriminator.evaluate(
                        np.concatenate([test_batch_x, generated_images]),
                        np.concatenate([real_label, fake_label]),
                        verbose=False
                    )
                else:
                    # loss_fake, acc_fake, *rest = \
                    #         self.discriminator_fake.evaluate([generated_images],
                    #                                         fake_label, verbose=False)
                    # loss_real, acc_real, *rest = \
                    #         self.discriminator_real.evaluate([test_batch_x],
                    #                                         real_label, verbose=False)
                    test_disc_loss = 0.5 * (1 + 1)

                negative_samples = bg_train.get_samples_by_labels(bg_train.other_labels(test_batch_y))
                # gen_d_loss, _ = self.combined.evaluate(
                #     [
                #         k_shot_test_batch,
                #         negative_samples,
                #         f
                #     ],
                #     real_label,
                #     verbose = 0
                # )
                gen_d_loss = 0

                if e % (50 // (self.resolution // 32)) == 0:
                    self.backup_point(e)
                    self.evaluate_d(
                        np.concatenate(X, axis=0),
                        np.concatenate(Y, axis=0))
                    self.evaluate_g(
                        [
                            k_shot_test_batch,
                            f,
                        ],
                        real_label,
                    )

                    crt_c = 0
                    # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c))

                    f = self.generate_latent([crt_c] * 10)
                    img_samples = np.array([
                        [
                            act_img_samples[:,:,:,:self.channels],
                            self.generate(act_img_samples, f)
                        ]
                    ])
                    for crt_c in range(1, min(self.nclasses, 3)):
                        # act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        act_img_samples = bg_train.ramdom_kshot_images(self.k_shot,
                                                                   np.full(10, crt_c))
                        f = self.generate_latent([crt_c] * 10)
                        new_samples = np.array([
                            [
                                act_img_samples[:,:,:,:self.channels],
                                self.generate(act_img_samples, f)
                            ]
                        ])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)

                    utils.show_samples(img_samples)

                    # calculate attribute distance
                    self.plot_loss_his()
                    # self.plot_feature_distr(bg_train)
                    per_class_size = 50 
                    classes = []
                    for i in bg_train.classes:
                        classes += [i] * per_class_size
                    for i in bg_test.classes:
                        classes += [i] * per_class_size

                    self.plot_cm_for_G(bg_train,
                                    bg_test,
                                    classes)

                if e % (100 // (self.resolution // 32)) == 0:
                    self.backup_point(e)

                print("- D_loss {}, G_loss {} - {}".format(
                    train_disc_loss, train_gen_loss,
                    datetime.datetime.now() - start_time
                ))

                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append(gen_d_loss)

            self.trained = True

    def plot_feature_distr(self, bg, size=50):
        x, y = bg.dataset_x, bg.dataset_y
        real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, bg.classes[0]))

        fakes = self.generate(real, self.generate_latent([0] * size))

        fake_labels = [np.full((size,), 'fake of 0')]

        for classid in bg.classes[1:5]:
            real = bg.ramdom_kshot_images(self.k_shot,
                                    np.full(size, classid))
            fake = self.generate(real, self.generate_latent([classid] * size))
            fakes = np.concatenate([fakes, fake])
            fake_labels.append(np.full((size,), 'fake of {}'.format(classid)))

        # latent_encoder
        imgs = np.concatenate([x, fakes])
        labels = np.concatenate([
            np.full((x.shape[0],), 'real'),
            np.full((fakes.shape[0],), 'fake'),
        ])

        try:
            utils.scatter_plot(imgs, labels, self.features_from_d_model, 'fake real space')
        except:
            print("can not draw fake real space")
        labels = np.concatenate([y, np.concatenate(fake_labels)])
        utils.scatter_plot(imgs, labels, self.latent_encoder, 'latent encoder')
