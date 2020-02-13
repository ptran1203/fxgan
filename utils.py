import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab.patches import cv2_imshow

DS_DIR = '/content/drive/My Drive/bagan/dataset/chest_xray'
DS_SAVE_DIR = '/content/drive/My Drive/bagan/dataset/save'

def pickle_save(object, path):
    with open(path, "wb") as f:
        return pickle.dump(object, f)


def pickle_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
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
    print('saving ', path)
    pickle_save(imgs, path)

def load_ds(rst, opt):
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    print('loading ', path)
    return pickle_load(path)

def save_image_array(img_array, fname):
    channels = img_array.shape[2]
    resolution = img_array.shape[-1]
    img_rows = img_array.shape[0]
    img_cols = img_array.shape[1]

    img = np.full([channels, resolution * img_rows, resolution * img_cols], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[:,
            (resolution * r): (resolution * (r + 1)),
            (resolution * (c % 10)): (resolution * ((c % 10) + 1))
            ] = img_array[r, c]

    img = ((img + 0.5) * 255).astype(np.uint8)
    if (img.shape[0] == 1):
        img = img[0]
    else:
        img = np.rollaxis(img, 0, 3)

    try:
        cv2_imshow(img)
    except Exception as e:
        print('[save fail] ', str(e))
        Image.fromarray(img).save(fname)

def path(path):
    return os.path.join(BASE_DIR, path)

def get_img(path, rst):
    img = cv2.imread(path)
    img = add_padding(img)
    img = cv2.resize(img, (rst, rst))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.tolist()

def load_train_data(s=0,  resolution=52):
    imgs = []
    labels = []
    i = 0
    res = load_ds(resolution, 'train')
    if res:
        return res
    for file in os.listdir(DS_DIR + '/train/NORMAL')[:s if s > 0 else 1349]:
        path = DS_DIR + '/train/NORMAL/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs.append(get_img(path, resolution))
            labels.append(0)
        except:
            pass

    for file in os.listdir(DS_DIR + '/train/PNEUMONIA')[:s if s > 0 else 3884]:
        path = DS_DIR + '/train/PNEUMONIA/' + file
        i += 1
        if i % 150 == 0:
            print(len(labels), end=',')
        try:
            imgs.append(get_img(path, resolution))
            labels.append(1)
        except:
            pass

    res = (np.array(imgs), np.array(labels))
    save_ds(res, resolution, 'train')
    return res

def load_test_data(s=0, resolution = 52):
    imgs = []
    labels = []
    res = load_ds(resolution, 'test')
    if res:
        return res
    for file in os.listdir(DS_DIR + '/test/NORMAL')[:s if s > 0 else 234]:
        path = DS_DIR + '/test/NORMAL/' + file
        try:
            imgs.append(get_img(path, resolution))
            labels.append(0)
        except:
            pass

    for file in os.listdir(DS_DIR + '/test/PNEUMONIA')[:s if s > 0 else 390]:
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

    def __init__(self, data_src, batch_size=5, class_to_prune=None, unbalance=0, dataset='MNIST', rst=64):
        self.batch_size = batch_size
        self.data_src = data_src

        # Load data
        if dataset == 'MNIST':
            mnist = input_data.read_data_sets("dataset/mnist", one_hot=False)

            assert self.batch_size > 0, 'Batch size has to be a positive integer!'

            if self.data_src == self.TEST:
                self.dataset_x = mnist.test.images
                self.dataset_y = mnist.test.labels
            else:
                self.dataset_x = mnist.train.images
                self.dataset_y = mnist.train.labels

            # Normalize between -1 and 1
            self.dataset_x = (np.reshape(self.dataset_x, (self.dataset_x.shape[0], 28, 28)) - 0.5) * 2

            # Include 1 single color channel
            self.dataset_x = np.expand_dims(self.dataset_x, axis=1)

        elif dataset == 'CIFAR10':
            ((x, y), (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()

            if self.data_src == self.TEST:
                self.dataset_x = x
                self.dataset_y = y
            else:
                self.dataset_x = x_test
                self.dataset_y = y_test

            # Arrange x: channel first
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 3, 1, 2))

            # Normalize between -1 and 1
            self.dataset_x = self.dataset_x/255 - 0.5

            # Y 1D format
            self.dataset_y = self.dataset_y[:, 0]
        else:
            if self.data_src == self.TEST:
                x, y = load_test_data(0, rst)
                self.dataset_x = x
                self.dataset_y = y
            else:
                x_test, y_test = load_train_data(0, rst)
                self.dataset_x = x_test
                self.dataset_y = y_test

            # Arrange x: channel first
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 1, 2))
            # Normalize between -1 and 1
            self.dataset_x = self.dataset_x/255 - 0.5
            self.dataset_x = np.expand_dims(self.dataset_x, axis=1)
            # Y 1D format
            # self.dataset_y = self.dataset_y[:, 0]

        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        # Prune if needed!
        if class_to_prune is not None:
            all_ids = list(np.arange(len(self.dataset_x)))

            mask = [class_to_prune == lc for lc in self.dataset_y]
            all_ids_c = np.array(all_ids)[mask]
            np.random.shuffle(all_ids_c)

            other_class_count = np.array(per_class_count)
            other_class_count = np.delete(other_class_count, class_to_prune)
            to_keep = int(np.ceil(unbalance * max(
                other_class_count)))

            to_delete = all_ids_c[to_keep: len(all_ids_c)]

            self.dataset_x = np.delete(self.dataset_x, to_delete, axis=0)
            self.dataset_y = np.delete(self.dataset_y, to_delete, axis=0)

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
