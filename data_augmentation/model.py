from keras.models import Model, load_model
from keras import applications
import keras.backend as K
import matplotlib.pyplot as plt

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
from classification_models.keras import Classifiers
from mlxtend.plotting import plot_confusion_matrix
from sklearn.utils import class_weight as sk_weight

from keras.layers.convolutional import (
    UpSampling2D,
    Conv2D, Conv2DTranspose
)
from keras.optimizers import Adam, SGD

from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import keras.preprocessing.image as iprocess
import sklearn.metrics as sk_metrics
from utils import *
import triplet_loss
from data_augmentation.data_loader import load_gen

classifier = None
train_model = None
model_map = [0, 'GAN v1', 'GAN v2', 'BAGAN',
             'VGG16', 'VGG16 + standard augment']

class Option:
    gan_v1 = 1
    gan_v2 = 2
    bagan = 3
    vgg16 = 4
    vgg16_st_aug = 5

class Losses:
    center = 1
    triplet = 2

def get_pretrained_model(name, input_shape, weights):
    """
    name should be: [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'seresnet18',
        'seresnet34',
        'seresnet50',
        'seresnet101',
        'seresnet152',
        'seresnext50',
        'seresnext101',
        'senet154',
        'resnet50v2',
        'resnet101v2',
        'resnet152v2',
        'resnext50',
        'resnext101',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'densenet201',
        'inceptionresnetv2',
        'inceptionv3',
        'xception',
        'nasnetlarge',
        'nasnetmobile',
        'mobilenet',
        'mobilenetv2'
    ]
    """

    model, _ = Classifiers.get(name)
    return model(input_shape=input_shape,
                 weights=weights,
                 include_top=False)

def plot_history(losses):
      # plot loss
    plt.plot(losses, label='train')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training loss')
    plt.legend()
    plt.show()

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

def feature_extractor(image, num_of_classes,
                    dims=64, rst=64,
                    from_scratch=True,
                    frozen_block=[],
                    name='vgg16',
                    loss_type=Losses.center):
    weights = None if from_scratch else 'imagenet'

    model = get_pretrained_model(name=name,
                                input_shape=(rst, rst, 3),
                                weights=weights,
                                )

    for layer in model.layers:
        if any(x in layer.name for x in frozen_block):
            layer.trainable = False
        else:
            layer.trainable = True

    x = model(image)
    x = GlobalAveragePooling2D()(x)
    x = Dense(dims)(x)
    if loss_type == Losses.center:
        out1 = keras.layers.advanced_activations.PReLU(name='side_out')(x)
        out2 = Dense(num_of_classes, activation='softmax', name='main_out')(out1)
        return out1, out2

    return x


def main_model(num_of_classes, rst=64, feat_dims=128, lr=1e-5,
                loss_weights=[1, 0.1],
                from_scratch=True,frozen_block=[],
                name='vgg16',decay=None,loss_type=Losses.center):
    images = Input((rst, rst, 3))
    labels = Input((1,))
    outputs = feature_extractor(images,
                                num_of_classes,
                                feat_dims,
                                rst,
                                from_scratch,
                                frozen_block=frozen_block,
                                name=name,loss_type=loss_type)
    optimizer = Adam(lr, decay=decay) if decay else Adam(lr)

    if loss_type == Losses.center:
        side_output, final_output = outputs
        centers = Embedding(num_of_classes, feat_dims)(labels)
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),
                            name='l2_loss')([side_output ,centers])

        labels_plus_embeddings = Concatenate()([labels, side_output])
        train_model = Model(inputs=[images, labels],
                            # outputs=labels_plus_embeddings,
                            outputs=[final_output, l2_loss]
                            )
        train_model.compile(optimizer=optimizer,
                            loss=["categorical_crossentropy",lambda y_true,y_pred: y_pred],
                            # loss = triplet_loss_adapted_from_tf,
                            loss_weights=loss_weights,
                            metrics=['accuracy'])
    else:
        # https://github.com/AdrianUng/keras-triplet-loss-mnist
        embeddings = outputs
        labels_plus_embeddings = Concatenate()([input_labels, embeddings])
        train_model = Model(inputs=[images, labels],
                            outputs=labels_plus_embeddings)
        train_model.compile(loss=triplet_loss,
                            optimizer=optimizer)

    return train_model
    


def l2_distance(a, b):
    return np.mean(np.square(a - b))

def cosine_sim(a, b):
    return - (np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def cal_sp_vectors(embbeder, supports,k_shot):
    means = []
    x_sp, y_sp = supports
    classes = np.unique(y_sp)
    # perclassid
    per_class_ids = dict()
    ids = np.array(range(len(x_sp)))
    for c in classes:
        per_class_ids[c] = ids[y_sp == c]

    for c in classes:
        imgs = x_sp[per_class_ids[c][:k_shot]]
        # imgs = utils.triple_channels(imgs)
        latent = embbeder.predict(imgs)
        means.append(np.mean(latent, axis=0))
    return np.array(means)
    
def classify_by_metric(embbeder, supports, images, k_shot=1,metric='l2'):
    x_sp, y_sp = supports
    classes = np.unique(y_sp)
    # currently do one-shot classification
    sp_vectors = cal_sp_vectors(embbeder, supports,k_shot)
    vectors = embbeder.predict(triple_channels(images))
    metric_func = l2_distance if metric == 'l2' else cosine_sim
    similiarity = np.array([metric_func(vector, sp_vector) \
                        for vector in vectors \
                        for sp_vector in sp_vectors]).reshape(-1, len(classes))
    pred = np.argmin(np.array(similiarity), axis=1)
    return pred

def evaluate_by_metric(embbeder, supports, images, labels, k_shot=1,metric='l2'):
    pred = classify_by_metric(embbeder, supports,
                              images,k_shot=k_shot, metric=metric)
    acc = (pred == labels).mean()
    return acc


def train_one_epoch(model, batch_gen, class_weight):
    total_loss = []
    for x, y in batch_gen.next_batch():
        loss_ = model.train_on_batch(
            x, y,
            class_weight=class_weight
        )
        total_loss.append(loss_[1])

    return np.mean(np.array(total_loss))


class BatchGen:
    """simple batch gen"""
    def __init__(self, x, y, batch_size=64, loss_type=Losses.center):
        self.x = triple_channels(x)
        self.y = y
        self.batch_size = batch_size
        self.num_of_classes = len(np.unique(y))
        self.dummy = np.zeros((self.batch_size, 129))
        self.loss_type = loss_type


    def next_batch(self):
        dataset_x = self.x
        labels = self.y
        onehot_labels = to_categorical(labels, self.num_of_classes)

        indices = np.arange(dataset_x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]

            if self.loss_type == Losses.center:
                batch_y = [onehot_labels[access_pattern], self.dummy]
            else:
                batch_y = self.dummy
            yield (
                [dataset_x[access_pattern, :, :, :], labels[access_pattern]],
                batch_y,
            )


def save_embbeding(train_model, dataset='multi_chest', loss_type=Losses.center):
    embbeding_model = Model(
        inputs = train_model.inputs[0],
        outputs = train_model.get_layer('side_out').get_output_at(-1)
    ) if loss_type == Losses.center else train_model

    fname = '/content/drive/My Drive/bagan/{}/latent_encoder_{}'.format(dataset, x_train.shape[1])
    with open(fname + '.json', 'w', encoding='utf-8') as f:
        print('Save json model')
        f.write(embbeding_model.to_json())
    embbeding_model.save(fname + '.h5')
    print("Save model for dataset ", dataset)


def confusion_mt(model, test_x, test_y):
    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)
    cm = sk_metrics.confusion_matrix(y_true=test_y, y_pred=y_pred)
    plt.figure()
    plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
    plt.show()

def shuffle_data(data_x, data_y):
    rd_idx = np.arange(data_x.shape[0])
    np.random.shuffle(rd_idx)
    return data_x[rd_idx], data_y[rd_idx]


def evaluate_model(train_model, x_test, y_test,
                   y_test_onehot):

    x_test_3 = triple_channels(x_test)
    classifier = Model(inputs = train_model.inputs[0],
                        outputs = train_model.get_layer('main_out').get_output_at(-1))
    classifier.compile(optimizer='adam', metrics = ['accuracy'],
                        loss='categorical_crossentropy')
    accuracy = classifier.evaluate(x_test_3, y_test_onehot, verbose=0)[1]
    auc = metrics.auc_score(y_test, classifier.predict(x_test_3), verbose=0)
    confusion_mt(classifier, x_test_3, y_test)
    return accuracy, auc

def evaluate_model_metric(embbeder, supports, x_test, y_test ,k_shot=1, metric='l2'):
    x_test_3 = triple_channels(x_test)

    y_pred = classify_by_metric(embbeder, supports,
                              x_test_3, k_shot=k_shot,
                              metric=metric)

    cm = sk_metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    plt.figure()
    plot_confusion_matrix(cm, hide_ticks=True,cmap=plt.cm.Blues)
    plt.show()
    return (pred == labels).mean(), 0

## ==== Run training ==== ##
def run(mode, x_train, y_train, test_data ,experiments = 1, frozen_block=[],
        name='vgg16', save=False, lr=1e-5,
        loss_weights=[1, 0.1], epochs=25, loss_type=Losses.center, lr_decay=None,
        k_shot=1):

    x_test, y_test = test_data
    class_counter = dict(Counter(y_train))
    classes = np.unique(y_train)
    num_of_classes = len(classes)

    
    if experiments > 1:
        # only use k_shot images in useen classes (pneumonia, herina)
        keep = [0] * 13
        to_remove = [
            class_counter[13] - k_shot,
            class_counter[14] - k_shot,
        ]
        x_train, y_train = prune(x_train, y_train, keep+to_remove)
        


    class_weight = sk_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    class_weight = dict(enumerate(class_weight))
    if loss_type == Losses.triplet:
        class_weight  =  None
    if mode == 4:
        x_train_aug, y_train_aug = x_train, y_train
    elif mode == 5:
        x_train_aug, y_train_aug = re_balance(
            x_train,
            y_train,
            [(class_counter[1] - class_counter[i]) + 0 for i in range(num_of_classes)])
    else:
        print("Train on fake data")
        x_train_aug, y_train_aug = load_gen('multi_chest', mode)
        # to balance
        counter = dict(Counter(y_train))
        prune_classes = [max(2000 - (2183 - counter[cls]), 0) for cls in classes]
        x_train_aug, y_train_aug = prune(x_train_aug, y_train_aug, [])

        print("Augment data: ", Counter(y_train_aug))
        print("Origin data: ", Counter(y_train))

        if mode == Option.bagan:
            x_train_aug = x_train_aug *127.5+127.5
        show_samples(x_train_aug[:10])

        x_train_aug, y_train_aug = (np.concatenate([x_train,x_train_aug]),
                                    np.concatenate([y_train, y_train_aug]))
        x_train_aug = triple_channels(x_train_aug)

    # run 5 experiments
    acc = []
    auc_scores = []
    batch_size = 128
    print("learning rate decay ", lr_decay)
    print(Counter(y_train_aug))
    batch_gen = BatchGen(x_train_aug, y_train_aug, 128, loss_type=loss_type)
    for i in range(experiments):
        print("run experiments {}/{} - {}".format(i + 1, experiments, model_map[mode]))
        train_model = main_model(num_of_classes, 128,
                            128, lr=lr,
                            loss_weights=loss_weights,
                            from_scratch=False,
                            frozen_block=[],
                            name=name,
                            decay=lr_decay,
                            loss_type=loss_type)
        
        losses = []
        for i in range(epochs):
            start_time = datetime.datetime.now()
            loss_mean = train_one_epoch(train_model, batch_gen, class_weight)
            print("epochs {}/{} - loss: {} - {}".format(
                i + 1, epochs, loss_mean, datetime.datetime.now() - start_time
            ))
            losses.append(loss_mean)


        if save:
            save_embbeding(train_model, dataset, loss_type=loss_type)

        if loss_type == Losses.center:
            y_test_onehot = to_categorical(y_test, num_of_classes)
            accuracy, auc = evaluate_model(train_model, x_test, y_test, y_test_onehot)
        else:
            x_test_u, x_sp_u, y_test_u, y_sp_u = train_test_split(x_test, y_test)
            accuracy, auc = evaluate_model_metric(train_model,
                                        ( x_sp_u, y_sp_u), 
                                        x_test_u, y_test_u ,
                                        k_shot=1, metric='l2')
        print("Acc ", accuracy)
        print("Auc ", auc)
        acc.append(accuracy)
        auc_scores.append(auc)

    ## calculate avg
    mean_acc = np.mean(np.array(acc))
    mean_auc = np.mean(np.array(auc_scores), axis=0)
    return mean_auc

