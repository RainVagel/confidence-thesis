import numpy as np
from sklearn import datasets
import tensorflow as tf


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


class CifarDataset:
    def __init__(self, cifar_version=10):
        self.cifar_version = cifar_version

    def load_dataset(self):
        if self.cifar_version == 10:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif self.cifar_version == 100:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            raise Exception("Unsupported CIFAR version: {}".format(self.cifar_version))

        y_train = tf.keras.utils.to_categorical(y_train, self.cifar_version)
        y_test = tf.keras.utils.to_categorical(y_test, self.cifar_version)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test

    def load_label_names(self):
        if self.cifar_version == 10:
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise Exception("Unsupported CIFAR version for labels: {}".format(self.cifar_version))


class MnistDataset:
    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        X_train = x_train / 255.
        X_test = x_test / 255.

        x_train = np.reshape(X_train, (60000, 28, 28, 1))
        x_test = np.reshape(X_test, (10000, 28, 28, 1))

        return x_train, y_train, x_test, y_test