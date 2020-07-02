from keras.models import Model, load_model
from keras import applications
import keras.backend as K
import keras
from keras_contrib.applications.resnet import ResNet, basic_block
from keras_contrib.applications.densenet import DenseNet
import keras.applications as k_apps
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Embedding, Dropout,
    BatchNormalization, Activation,
    Lambda,Layer, Add, Concatenate,
    Average,GlobalAveragePooling2D,
    MaxPooling2D, AveragePooling2D,
)
from keras.layers.convolutional import (
    UpSampling2D,
    Conv2D, Conv2DTranspose
)
from keras.optimizers import Adam, SGD

from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import keras.preprocessing.image as iprocess



def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)

    model_flat = keras.models.Sequential(layers_flat)
    return model_flat

def tran_one(img, d=15):
    degree = np.random.randint(-d, d)
    img = iprocess.random_brightness(img, (0.5, 1.5))
    if np.random.rand() >= 0.5:
       img = np.fliplr(img)
 
    return iprocess.apply_affine_transform(img, degree)

def augment(imgs, labels,plus = 1, target_labels=None):
    if plus == 0:
        return imgs, labels
    if target_labels is None:
        target_labels = np.unique(labels)

    deimgs = imgs *127.5 + 127.5
    imgs_ = []
    labels_ = []
    for i in range(imgs.shape[0]):
        # original
        imgs_.append(deimgs[i])
        labels_.append(labels[i])
        if labels[i] not in target_labels:
            continue

        # Augment
        for j in range(plus):
            imgs_.append(tran_one(deimgs[i]))
            labels_.append(labels[i])

    return ((np.array(imgs_) -127.5) / 127.5), np.array(labels_)


def re_balance(imgs, labels, per_class_samples=None):
    deimgs = imgs *127.5 + 127.5
    imgs_ = []
    labels_ = []
    size = len(np.unique(labels))
    counter = [0] * size
    if per_class_samples is None:
        per_class_samples = [1000] * size

    print(counter)
    print(per_class_samples)
    # original
    for i in range(imgs.shape[0]):
        
        imgs_.append(deimgs[i])
        labels_.append(labels[i])
    # Augment
    for _ in range(1000):
        for i in range(imgs.shape[0]):
            l_idx = labels[i]
            if counter[l_idx] >= per_class_samples[l_idx]:
                counter[l_idx] = -1 # Done
            if counter[l_idx] != -1:
                imgs_.append(tran_one(deimgs[i]))
                labels_.append(l_idx)
                counter[l_idx] += 1

        print(counter)
        print(counter.count(-1), size)
        if counter.count(-1) == size:
            break

    return ((np.array(imgs_) -127.5) / 127.5), np.array(labels_)

def vgg_16_features(image, num_of_classes, dims=64, rst=64):
    model = k_apps.VGG16(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(rst, rst, 3),
                        pooling='avg',
                        classes=num_of_classes)


    for layer in model.layers:
        accept_name = ['block1', 'block2', 'block3', 'block4', 'block5'][:1]
        if any(x in layer.name for x in accept_name):
            layer.trainable = False
        else:
            layer.trainable = True

    x = model(image)
    x = Dense(dims)(x)
    out1 = keras.layers.advanced_activations.PReLU(name='side_out')(x)
    out2 = Dense(num_of_classes, activation='softmax', name='main_out')(out1)
    return out1, out2

def main_model(num_of_classes, rst=64, feat_dims=128, lr=1e-5, loss_weights=[1, 0.1]):
    image = Input((rst, rst, 3))
    labels = Input((1,))
    side_output, final_output = vgg_16_features(image, num_of_classes, feat_dims, rst)

    centers = Embedding(num_of_classes, feat_dims)(labels)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),
                        name='l2_loss')([side_output ,centers])

    labels_plus_embeddings = Concatenate()([labels, side_output])
    train_model = Model(inputs=[image, labels],
                        # outputs=labels_plus_embeddings,
                        outputs=[final_output, l2_loss]
                        )
    train_model.compile(optimizer=Adam(lr),
                                loss=["categorical_crossentropy",lambda y_true,y_pred: y_pred],
                                # loss = triplet_loss_adapted_from_tf,
                                loss_weights=loss_weights,
                                metrics=['accuracy'])

    return train_model
