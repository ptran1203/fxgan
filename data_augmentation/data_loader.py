import numpy as np
import pickle
from const import BASE_DIR, INVERT_CATEGORIES_MAP, CATEGORIES_MAP
from utils import *
from collections import Counter
from sklearn.model_selection import train_test_split


def load_gen(ds_name, version=1):
    x,y = pickle_load(
        '/content/drive/My Drive/generated/{}/imgs_train_gen_v{}.pkl'.format(ds_name, version)
    )
    x = (x - 127.5) / 127.5
    return x,y


def _load_multi_chest(rst, large, classes):
    file_name = 'imgs_labels'
    if not large:
        file_name += '_' + str(rst)
    file_name += '.pkl'
    x, y = pickle_load(BASE_DIR + '/dataset/multi_chest/'  + file_name)
    to_keep = [i for i, l in enumerate(y) if '|' not in l]
    to_keep = np.array(to_keep)
    x = x[to_keep]
    y = y[to_keep]
    to_train_classes = INVERT_CATEGORIES_MAP[:classes]
    to_keep = np.array([i for i, l in enumerate(y) if l in to_train_classes])
    x_train, y_train = x[to_keep], y[to_keep]
    y_train = np.array([CATEGORIES_MAP[l] for l in y_train])

    to_keep = np.array([i for i, l in enumerate(y) if l not in to_train_classes])

    # unseen classes data
    if len(to_keep) > 0:
        x_unseen, y_unseen = x[to_keep], y[to_keep]
        x_unseen = (x_unseen - 127.5) / 127.5
        y_unseen = np.array([CATEGORIES_MAP[l] for l in y_unseen])
    else:
        x_unseen, y_unseen = None, None

    return x_train, y_train, x_unseen, y_unseen

def _load_chest(rst):
    x_train, y_train  = load_ds(rst, 'train')
    x_test, y_test  = load_ds(rst, 'test')
    x_test = (x_test - 127.5) / 127.5
    x_test = triple_channels(x_test)
    x_train, y_train = prune(x_train, y_train, [0, 2400])
    return x_train, y_train, x_test, y_test

def _load_flower(classes):
    x, y = pickle_load(BASE_DIR + '/dataset/flowers/imgs_labels.pkl')
    y = y - 1
    to_train_classes = list(range(0, classes))
    to_test_classes = list(range(80, 85))
    to_keep = np.array([i for i, l in enumerate(y) if l in to_test_classes])
    x_unseen, y_unseen = x[to_keep], y[to_keep]
    to_keep = np.array([i for i, l in enumerate(y) if l in to_train_classes])
    x_train, y_train = x[to_keep], y[to_keep]
    x_unseen = (x_unseen - 127.5) / 127.5

    return x_train, y_train, x_unseen, y_unseen

def load_dataset(dataset='multi_chest',
                resolution=64,
                large=False,
                classes=5,
                test_val_split=[0.3, 0.1]):
    """
    return: train_pair, val_pair, test_pair, unseen_pair
    """
    if dataset == 'multi_chest':
        x_train, y_train, x_unseen, y_unseen = _load_multi_chest(resolution, large, classes)

    elif dataset == 'chest':
        x_train, y_train, x_test, y_test = _load_chest(resolution)
        x_unseen, y_unseen = None, None
    elif dataset == 'flowers':
        x_train, y_train, x_unseen, y_unseen = _load_flower(classes)

    # norm
    x_train = (x_train - 127.5) / 127.5
    random = np.arange(x_train.shape[0])
    np.random.shuffle(random)
    x_train, y_train = x_train[random], y_train[random]
    # x_train = triple_channels(x_train)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=test_val_split[1])
    if dataset != 'chest':
        x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                y_train,
                                                test_size=test_val_split[0])
    # if x_unseen is not None:
        # x_unseen = triple_channels(x_unseen)

    print("\n===== data loaded =====\n")
    print("TRAIN: ", Counter(y_train))
    return (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        x_unseen, y_unseen
    )
