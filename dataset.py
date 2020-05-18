import math

import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random
import emnist
import scipy.io as spio
import torch
import torchvision.transforms as transforms
import torchvision as tv


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
                    if self.dim == 8:
                        if x_i == 1 and y_i == 0:
                            y_extra = [2 if i == 1 else 3 for i in y_extra]
                            y_test_extra = [2 if i == 1 else 3 for i in y_test_extra]
                        if x_i == 0 and y_i == 1:
                            y_extra = [4 if i == 1 else 5 for i in y_extra]
                            y_test_extra = [4 if i == 1 else 5 for i in y_test_extra]
                        if x_i == 1 and y_i == 1:
                            y_extra = [6 if i == 1 else 7 for i in y_extra]
                            y_test_extra = [6 if i == 1 else 7 for i in y_test_extra]
                    X = np.append(X, X_extra, axis=0)
                    y = np.append(y, y_extra, axis=0)
                    X_test = np.append(X_test, X_test_extra, axis=0)
                    y_test = np.append(y_test, y_test_extra, axis=0)
        y = tf.one_hot(y, self.dim)
        X = tf.convert_to_tensor(X)
        return X, y, X_test, y_test
