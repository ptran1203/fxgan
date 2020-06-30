
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import keras.backend as K
import os
import re
import numpy as np
import datetime
import pickle
import cv2
import urllib.request


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
    if image.shape[-1] == 3:
        return image

    return np.repeat(image, 3, axis = -1)

def pickle_save(object, path):
    try:
        print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))


def http_get_img(url, rst=64, gray=False):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.resize(img, (rst, rst))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img.reshape((1, rst, rst, -1)) - 127.5) / 127.5

def save_weights(model, dir):
    name = model.name + '.h5'
    weights = [l.get_weights() for l in model.layers]
    pickle_save(weights, os.path.join(dir, name))

def set_weights(model, dir):
    name = model.name + '.h5'
    weights = pickle_load(os.path.join(dir, name))
    for layer, w in zip(model.layers, weights):
        layer.set_weights(w)


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

def visualize_scatter_with_images(X_2d_data, images, figsize=(10,10), image_zoom=0.5):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()

def visualize_scatter(data_2d, label_ids, figsize=(8,8)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for i,label_id in enumerate(np.unique(label_ids)):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(i / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=label_id)
    plt.legend(loc='best')
    plt.show()

def scatter_plot(x, y, encoder, name='chart', opt='pca', plot_img=None):
    step = 1
    if encoder.input_shape[-1] != x.shape[-1]:
        x = triple_channels(x)

    x_embeddings = encoder.predict(x)
    if len(x_embeddings.shape) > 2:
        x_embeddings = x_embeddings.reshape(x_embeddings.shape[0], -1)
    decomposed_embeddings = decomposers[opt].fit_transform(x_embeddings)
    if plot_img:
        return visualize_scatter_with_images(decomposed_embeddings,x)
    visualize_scatter(decomposed_embeddings, y)


def plot_model_history(H, opt = 0):
    """
    opt = 0: plot both
    opt = 1: plot loss
    opt = 2: plot acc
    """
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

def group_k_images(image_list):
    return np.array(image_list)


def prune(x, y, prune_classes):
    """
    prune data by give classes
    """
    for class_to_prune in range(len(prune_classes)):
        print(class_to_prune)
        remove_size = prune_classes[class_to_prune]
        all_ids = list(np.arange(len(x)))
        mask = [lc == class_to_prune for lc in y]
        all_ids_c = np.array(all_ids)[mask]
        np.random.shuffle(all_ids_c)
        to_delete  = all_ids_c[:remove_size]
        x = np.delete(x, to_delete, axis=0)
        y = np.delete(y, to_delete, axis=0)
        print('Remove {} items in class {}'.format(remove_size, class_to_prune))
    return x, y


def get_dataset(dataset, resolution, prune):
    pass