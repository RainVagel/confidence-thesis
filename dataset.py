import math

import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random
import emnist
import scipy.io as spio


class Dataset:
    def __init__(self, aug=True):
        self.aug = aug

    def load_dataset(self):
        pass


def get_cut_parameters(image_x, image_y, output_x, output_y):
    if image_x == output_x and image_y == output_y:
        return 0, 0, image_x, image_y

    i = random.randint(0, image_x - output_x)
    j = random.randint(0, image_y - output_y)

    return i, j, output_x, output_y


def crop_image(img, image_x, image_y, output_x, output_y):
    i, j, h, w = get_cut_parameters(image_x, image_y, output_x, output_y)
    return tf.image.crop_to_bounding_box(img, i, j, h, w)


class MoonsDataset:
    def __init__(self, dim=2):
        self.dim = dim

    def _generate_moons(self, n_samples=2000, shuffle=True, noise=.02, x_lat_mult=0, x_long_mult=0):
        X, y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise)
        # Rescale and shift the dataset to better fit into zero-one box
        X = (X + 1.6) / 4
        X[:, 0] = X[:, 0] - 0.035
        X[:, 1] = (X[:, 1] - 0.17) * 1.75
        X[:, 0] = X[:, 0] + x_lat_mult
        X[:, 1] = X[:, 1] + x_long_mult
        return X, y

    def two_moons(self, n_samples=2000, shuffle=True, noise=.02,
                  testing_samples=400, testing_shuffle=True, testing_noise=.02):
        X, y = self._generate_moons(n_samples, shuffle, noise)
        y = tf.one_hot(y, self.dim)
        X = tf.convert_to_tensor(X)
        X_test, y_test = self._generate_moons(testing_samples, testing_shuffle, testing_noise)
        return X, y, X_test, y_test

    def two_set_two_moons(self, n_samples=2000, shuffle=True, noise=.02,
                          testing_samples=400, testing_shuffle=True, testing_noise=.02, position='horizontal'):
        X, y = self._generate_moons(n_samples, shuffle, noise)

        X_test, y_test = self._generate_moons(testing_samples, testing_shuffle, testing_noise)

        if position == 'horizontal':
            X_extra, y_extra = self._generate_moons(n_samples, shuffle, noise, 1, 0)
            X_test_extra, y_test_extra = self._generate_moons(testing_samples, testing_shuffle, testing_noise, 1, 0)
        elif position == 'vertical':
            X_extra, y_extra = self._generate_moons(n_samples, shuffle, noise, 0, 1)
            X_test_extra, y_test_extra = self._generate_moons(testing_samples, testing_shuffle, testing_noise, 0, 1)
        else:
            raise Exception("Unknown two moons command")

        X = np.append(X, X_extra, axis=0)
        y = np.append(y, y_extra, axis=0)
        X_test = np.append(X_test, X_test_extra, axis=0)
        y_test = np.append(y_test, y_test_extra, axis=0)

        y = tf.one_hot(y, self.dim)
        X = tf.convert_to_tensor(X)

        return X, y, X_test, y_test

    def four_set_two_moons(self, n_samples=2000, shuffle=True, noise=.02,
                           testing_samples=400, testing_shuffle=True, testing_noise=.02):
        X, y = self._generate_moons(n_samples, shuffle, noise)

        X_test, y_test = self._generate_moons(testing_samples, testing_shuffle, testing_noise)

        for x_i in (0, 1):
            for y_i in (0, 1):
                if x_i == 0 and y_i == 0:
                    continue
                else:
                    X_extra, y_extra = self._generate_moons(n_samples, shuffle, noise, x_i, y_i)
                    X_test_extra, y_test_extra = self._generate_moons(testing_samples, testing_shuffle,
                                                                      testing_noise, x_i, y_i)
                    X = np.append(X, X_extra, axis=0)
                    y = np.append(y, y_extra, axis=0)
                    X_test = np.append(X_test, X_test_extra, axis=0)
                    y_test = np.append(y_test, y_test_extra, axis=0)
        y = tf.one_hot(y, self.dim)
        X = tf.convert_to_tensor(X)
        return X, y, X_test, y_test


class CifarDataset(Dataset):
    def __init__(self, aug=True, cifar_version=10):
        super().__init__(aug)
        self.cifar_version = cifar_version
        self.n_train, self.n_test = 50000, 10000
        self.n_classes = cifar_version
        self.height, self.width, self.n_colors = 32, 32, 3

    def load_dataset(self):
        if self.n_classes == 10:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif self.n_classes == 100:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            raise Exception("Unsupported CIFAR version: {}".format(self.n_classes))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test

    def load_label_names(self):
        if self.cifar_version == 10:
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise Exception("Unsupported CIFAR version for labels: {}".format(self.n_classes))


class Cifar10GrayScale(Dataset):

    def __init__(self, aug=True):
        super().__init__(aug)
        self.n_train, self.n_test = 50000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 3

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        #y_train = tf.keras.utils.to_categorical(y_train, self.n_classes)
        #y_test = tf.keras.utils.to_categorical(y_test, self.n_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Rehsaping to correct dimensionality
        #x_train = np.reshape(x_train, (self.n_train, 32, 32, 1))
        #x_test = np.reshape(x_test, (self.n_test, 32, 32, 1))

        x_train = tf.image.rgb_to_grayscale(x_train)
        x_test = tf.image.rgb_to_grayscale(x_test)

        # Resizing to a smaller size
        x_train = tf.image.resize(x_train, size=(self.height, self.width))
        x_test = tf.image.resize(x_test, size=(self.height, self.width))

        #if self.aug:
        #    x_train = tf.image.random_flip_left_right(x_train)
        #    x_train = tf.image.resize_with_pad(x_train, self.height + 8, self.width + 8)
        #    x_train = tf.map_fn(lambda x: crop_image(x, self.height + 4, self.width + 4, self.height, self.width),
        #                        x_train, dtype=tf.float32)

        # Normalizing and making to grayscale
        #x_train /= 255.0
        #x_test /= 255.0

        return x_train, y_train, x_test, y_test


class MnistDataset(Dataset):
    def __init__(self, aug=True):
        super().__init__(aug)
        self.n_train, self.n_test = 60000, 10000
        self.height, self.width, self.n_colors = 28, 28, 1
        self.n_classes = 10

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = np.reshape(x_train, (self.n_train, self.height, self.width, self.n_colors))
        x_test = np.reshape(x_test, (self.n_test, self.height, self.width, self.n_colors))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test


class FMnistDataset(Dataset):
    def __init__(self, aug=True):
        super().__init__(aug)
        self.n_train, self.n_test = 60000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 1

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        #y_train = tf.keras.utils.to_categorical(y_train, 10)
        #y_test = tf.keras.utils.to_categorical(y_test, 10)

        x_train = np.reshape(x_train, (self.n_train, self.height, self.width, self.n_colors))
        x_test = np.reshape(x_test, (self.n_test, self.height, self.width, self.n_colors))

        #if self.aug:
        #    x_train = tf.image.resize_with_pad(x_train, self.height + 8, self.width + 8)
        #    x_train = tf.map_fn(lambda x: crop_image(x, self.height + 4, self.width + 4, self.height, self.width),
        #                        x_train)

        #x_train = x_train / 255.
        #x_test = x_test / 255.

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test


class EMnistDataset(Dataset):
    def __init__(self, aug=True):
        super().__init__(aug)
        self.n_train, self.n_test = 88800, 14800
        self.n_classes = 37
        self.height, self.width, self.n_colors = 28, 28, 1

    def load_dataset(self):
        x_train, y_train = emnist.extract_training_samples('letters')
        x_test, y_test = emnist.extract_test_samples('letters')
        #y_train = tf.keras.utils.to_categorical(y_train, self.n_classes)
        #y_test = tf.keras.utils.to_categorical(y_test, self.n_classes)

        x_train, y_train, x_test, y_test = x_train[:self.n_train], y_train[:self.n_train], \
                                           x_test[:self.n_test], y_test[:self.n_test]

        x_train = np.reshape(x_train, (self.n_train, self.height, self.width, self.n_colors))
        x_test = np.reshape(x_test, (self.n_test, self.height, self.width, self.n_colors))

        #if self.aug:
        #    x_train = tf.image.resize_with_pad(x_train, self.height + 8, self.width + 8)
        #    x_train = tf.map_fn(lambda x: crop_image(x, self.height + 4, self.width + 4, self.height, self.width),
        #                        x_train, dtype=tf.float32)

        #x_train = x_train / 255.
        #x_test = x_test / 255.

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test


class SVHNDataset(Dataset):
    """
    training file link: http://ufldl.stanford.edu/housenumbers/train_32x32.mat
    testing file link: http://ufldl.stanford.edu/housenumbers/test_32x32.mat
    """

    def __init__(self, aug=True):
        super().__init__(aug)
        self.n_train, self.n_test = 73257, 26032
        self.n_classes = 10
        self.height, self.width, self.n_colors = 32, 32, 3
        self.path = 'svhn/'

    def _labeler(self, images):
        images[images == self.n_classes] = 0
        return images

    def _load_images(self):
        train_images = spio.loadmat(self.path + 'train_32x32.mat')
        test_images = spio.loadmat(self.path + 'test_32x32.mat')

        x_train = np.transpose(train_images['X'], (3, 0, 1, 2))

        y_train = self._labeler(train_images['y'])
        x_test = np.transpose(test_images['X'], (3, 0, 1, 2))
        y_test = self._labeler(test_images['y'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test

    def load_dataset(self):
        x_train, y_train, x_test, y_test = self._load_images()

        return x_train, y_train, x_test, y_test


class DataGenerator(Sequence):
    def __init__(self, dataset, batch_size, shuffle, mode='train', aug=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.aug = aug
        self.height = dataset.height
        self.width = dataset.width
        self.n_colors = dataset.n_colors
        self.n_classes = dataset.n_classes
        self.n_train = dataset.n_train
        self.n_test = dataset.n_test
        self.shuffle = shuffle
        self.x_train, self.y_train, self.x_test, self.y_test = self.dataset.load_dataset()
        if self.mode == 'train':
            self.indexes = np.arange(self.n_train)
        else:
            self.indexes = np.arange(self.n_test)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        if self.mode == 'train':
            return math.ceil(self.n_train/self.batch_size)
        else:
            return math.ceil(self.n_test/self.batch_size)

    def __getitem__(self, item):
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]

        X, y = self.__data_generation(indexes)
        return X, tf.keras.utils.to_categorical(y, self.n_classes)

    def get_analysis(self):
        indexes = self.indexes[:self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, tf.keras.utils.to_categorical(y, self.n_classes)

    def _random_crop(self, input_data):
        input_data = tf.image.resize_with_pad(input_data, self.height + 8, self.width + 8)
        input_data = tf.map_fn(lambda x: crop_image(x, self.height + 4, self.width + 4, self.height, self.width),
                            input_data, dtype=tf.float32)
        return input_data

    def _normalize_to_grayscale(self, input_data):
        # Normalizing and making to grayscale
        input_data = tf.map_fn(lambda x: np.array(x).mean(axis=2) / 255.0, input_data)
        return input_data

    @staticmethod
    def _random_horizontal_flip(input_data):
        input_data = tf.image.random_flip_left_right(input_data)
        return input_data

    @staticmethod
    def _normalize(input_data):
        input_data /= 255.
        return input_data

    def _reshape(self, input_data):
        input_data = np.reshape(input_data, (self.n_train, self.height, self.width, self.n_colors))
        return input_data

    def __data_generation(self, indexes):

        if self.mode == 'train':
            X = tf.gather(self.x_train, indexes, axis=0)
            y = tf.gather(self.y_train, indexes, axis=0)
        else:
            #X = tf.gather(x_test, indexes, axis=0)
            #y = tf.gather(y_test, indexes, axis=0)
            X = self.x_test
            y = self.y_test

        if self.aug is not None:
            for augmentation in self.aug:
                if augmentation.lower() == 'randomcrop':
                    X = self._random_crop(X)
                if augmentation.lower() == 'horizontalflip':
                    X = self._random_horizontal_flip(X)
                if augmentation.lower() == 'normalize':
                    X = self._normalize(X)
                if augmentation.lower() == 'reshape':
                    X = self._reshape(X)
                if augmentation.lower() == 'limitdataset':
                    x_train, y_train, x_test, y_test = self.x_train[:self.n_train], self.y_train[:self.n_train], \
                                                       self.x_test[:self.n_test], self.y_test[:self.n_test]
                if augmentation.lower() == 'normalizegreyscale':
                    X = self._normalize_to_grayscale(X)

        return X, y

    def on_epoch_end(self):
        if self.mode == 'train':
            self.indexes = np.arange(self.n_train)
        else:
            self.indexes = np.arange(self.n_test)
        if self.shuffle:
            np.random.shuffle(self.indexes)
