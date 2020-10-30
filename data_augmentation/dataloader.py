import numpy as np
import keras.preprocessing.image as image_processing
from utils import triple_channels
from keras.utils import to_categorical


class BatchGen:
    """simple batch gen"""

    def __init__(self, x, y, batch_size=64):
        self.x = triple_channels(x)
        self.y = y
        self.batch_size = batch_size
        self.num_of_classes = len(np.unique(y))
        self.dummy = np.zeros((self.batch_size, 129))

    def augment_one(self, x, y):
        new_x = _transform(x)
        return new_x, y

    def rebalance(self, x, y):
        """
        rebalance data in a batch
        """
        unique, counts = np.unique(y, return_counts=True)
        counter = dict(zip(unique, counts))
        most = max(counter.values())

        x_, y_ = [], []
        for i in range(len(y)):
            label = y[i]
            x_.append(x[i])
            y_.append(label)
            # augmentation
            if counter[label] < most:
                x_.append(_transform(x[i]))
                y_.append(label)
                counter[label] += 1

        return np.array(x_), np.array(y_)

    def next_batch(self):
        dataset_x = self.x
        labels = self.y
        onehot_labels = to_categorical(labels, self.num_of_classes)

        indices = np.arange(dataset_x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(
            0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size
        ):
            access_pattern = indices[start_idx : start_idx + self.batch_size]
            batch_y = [onehot_labels[access_pattern], self.dummy]

            balanced_x, balanced_y = self.rebalance(
                dataset_x[access_pattern, :, :, :], labels[access_pattern]
            )

            dummy = np.zeros((len(balanced_y), 129))
            one_hot = to_categorical(balanced_y, self.num_of_classes)
            yield (
                [balanced_x, balanced_y],
                [one_hot, dummy],
            )


def _transform(x):
    img = image_processing.random_rotation(x, 0.2)
    img = image_processing.random_shear(img, 30)
    img = image_processing.random_zoom(img, (0.5, 1.1))
    if np.random.rand() >= 0.5:
        img = np.fliplr(img)

    return img
