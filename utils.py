
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
import requests
import json
from const import BASE_DIR

try:
    from google.colab.patches import cv2_imshow
except:
    from cv2 import imshow as cv2_imshow

from PIL import Image

DS_SAVE_DIR = BASE_DIR + '/dataset/save'
MEAN_PIXCELS = np.array([103.939, 116.779, 123.68])


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

        img = denormalize(img).astype(np.uint8)
        if show:
            try:
                cv2_imshow(img)
            except Exception as e:
                fname = BASE_DIR + '/result/model_{}/img_{}.png'.format(
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
    return normalize(img.reshape((1, rst, rst, -1)))


def pickle_load(path):
    try:
        print("Loading data from {}".format(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
            print('load data successfully'.format(path))
            return data
    except Exception as e:
        print(str(e))
        return None


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


def visualize_scatter(data_2d, label_ids, figsize=(8,8), legend=True,title="None"):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))

    for i,label_id in enumerate(np.unique(label_ids)):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=colors[i],
                    linewidth='1',
                    alpha=0.8,
                    label=label_id)
    if legend:
        plt.legend(loc='best')
    else:
        # plt.title(title)
        plt.axis('off')

    plt.show()


def scatter_plot(x, y, encoder, name='chart', opt='pca', plot_img=None,
                legend=True, title="None"):
    step = 1
    if encoder.input_shape[-1] != x.shape[-1]:
        x = triple_channels(x)

    x_embeddings = encoder.predict(x)
    if len(x_embeddings.shape) > 2:
        x_embeddings = x_embeddings.reshape(x_embeddings.shape[0], -1)
    decomposed_embeddings = decomposers[opt].fit_transform(x_embeddings)
    if plot_img:
        return visualize_scatter_with_images(decomposed_embeddings,x)
    visualize_scatter(decomposed_embeddings, y, legend=legend,title=title)


def prune(x, y, prune_classes):
    """
    prune data by give classes
    """
    for class_to_prune in range(len(prune_classes)):
        remove_size = prune_classes[class_to_prune]
        if remove_size <= 0:
            continue
        print(class_to_prune)
        all_ids = list(np.arange(len(x)))
        mask = [lc == class_to_prune for lc in y]
        all_ids_c = np.array(all_ids)[mask]
        np.random.shuffle(all_ids_c)
        to_delete  = all_ids_c[:remove_size]
        x = np.delete(x, to_delete, axis=0)
        y = np.delete(y, to_delete, axis=0)
        print('Remove {} items in class {}'.format(remove_size, class_to_prune))
    return x, y


def preprocess(imgs):
    """
    BGR -> RBG then subtract the mean
    """
    return imgs - MEAN_PIXCELS


def deprocess(imgs):
    return imgs + MEAN_PIXCELS


def normalize(imgs):
    return ((imgs) - 127.5) / 127.5

def denormalize(imgs):
    return (imgs * 127.5 + 127.5)


def load_chestxray14_data(rst=128):
    return pickle_load(os.path.join(
        BASE_DIR,
        'dataset/multi_chest/imgs_labels_{}.pkl'.format(rst)
    ))