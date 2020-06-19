
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import keras.backend as K
import os
import re
import numpy as np
import datetime
import pickle
import cv2

from google.colab.patches import cv2_imshow
from PIL import Image
DS_SAVE_DIR = '/content/drive/My Drive/bagan/dataset/save'
DS_DIR = '/content/drive/My Drive/bagan/dataset/chest_xray'

decomposers = {
    'pca': PCA(),
    'tsne': TSNE()
}

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

def plot_data_space(x, y, encoder, name, opt='pca'):
    step = 1
    x_embeddings = encoder.predict(x)
    if len(x_embeddings.shape) > 2:
        x_embeddings = x_embeddings.reshape(x_embeddings.shape[0], -1)
    decomposed_embeddings = decomposers[opt].fit_transform(x_embeddings)
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


def plot_model_history(H):
    # plot loss
    plt.figure(figsize=(8,8))
    for k, loss in H.history.items():
        if 'loss' in k and 'acc' not in k:
            plt.plot(H.history[k], label=k)

    plt.legend()
    plt.title('Training loss')
    plt.show()

    # accuracy
    c = 0
    for k, acc in H.history.items():
        if 'acc' in k:
            c += 1
            plt.plot(H.history[k], label=k)

    if c > 0:
        plt.legend()
        plt.title('Training acc')
        plt.show()


def visualize_class_activation_map(model, image):
    width, height, _ = image.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(image), (2, 0, 1))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]

    final_conv_layer = model.layers[-2]

    get_output = K.function([model.layers[0].input], \
                            [final_conv_layer.output, 
                            model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    target_class = 1
    for i, w in enumerate(class_weights[:, target_class]):
            cam += w * conv_outputs[i, :, :]