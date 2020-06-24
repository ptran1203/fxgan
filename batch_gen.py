import utils
import numpy as np
from collections import Counter
from const import CATEGORIES_MAP, INVERT_CATEGORIES_MAP



class BatchGenerator:
    TRAIN = 1
    TEST = 0
    D_SIZE = 400
    to_train_classes = list(range(1, 80))
    to_test_classes = list(range(81, 86))

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

        if dataset == 'chest':
            if self.data_src == self.TEST:
                x, y = utils.load_test_data(rst)
                self.dataset_x = x
                self.dataset_y = y

            else:
                x, y = utils.load_train_data(rst)
                self.dataset_x = x  
                self.dataset_y = y

        elif dataset == 'flowers':
            x, y = utils.pickle_load('/content/drive/My Drive/bagan/dataset/flowers/imgs_labels.pkl')
            to_train_classes = self.to_train_classes
            to_test_classes = self.to_test_classes

            if self.data_src == self.TEST:
                to_keep = np.array([i for i, l in enumerate(y) if l in to_test_classes])
                x, y = x[to_keep], y[to_keep]
                self.dataset_x = x
                # TODO start from 0
                self.dataset_y = y - 1
            else:
                to_keep = np.array([i for i, l in enumerate(y) if l in to_train_classes])
                x, y = x[to_keep], y[to_keep]
                self.dataset_x = x
                # TODO start from 0
                self.dataset_y = y - 1


        else: # multi chest
            x, y = utils.pickle_load('/content/drive/My Drive/bagan/dataset/multi_chest/imgs_labels_{}.pkl'.format(rst))
            to_train_classes = self.to_train_classes
            to_test_classes = self.to_test_classes

            to_keep = [i for i, l in enumerate(y) if '|' not in l]
            to_keep = np.array(to_keep)
            x = x[to_keep]
            y = y[to_keep]
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
        for c in classes:
            self.per_class_ids[c] = ids[self.labels == c]


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
        # np.random.shuffle(indices2)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            # access_pattern2 = indices2[start_idx:start_idx + self.batch_size]

            yield (
                dataset_x[access_pattern, :, :, :], labels[access_pattern],
                # dataset_x[access_pattern2, :, :, :], labels[access_pattern2]
            )

    def ramdom_kshot_images(self, k_shot, labels, triple=True):
        imgs = []
        for label in labels:
            np.random.shuffle(self.per_class_ids[label])
            ids = self.per_class_ids[label][:k_shot]
            imgs.append(self.dataset_x[ids])
        imgs = np.array(imgs)

        if triple:
            imgs = utils.triple_channels(imgs)
        return imgs

