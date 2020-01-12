from sklearn import datasets
import tensorflow as tf


class CreateDataset:
    def __init__(self, dim=2):
        self.dim = dim

    def two_moons(self, n_samples=2000, shuffle=True, noise=.02,
                  testing_samples=400, testing_shuffle=True, testing_noise=.02):
        X, y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise)
        # Rescale and shift the dataset to better fit into zero-one box
        X = (X + 1.6) / 4
        X[:, 0] = X[:, 0] - 0.035
        X[:, 1] = (X[:, 1] - 0.17) * 1.75
        y = tf.one_hot(y, self.dim)

        X_test, y_test = datasets.make_moons(n_samples=testing_samples, shuffle=testing_shuffle, noise=testing_noise)
        # Rescale and shift the dataset to better fit into zero-one box
        X_test = (X_test + 1.6) / 4
        X_test[:, 0] = X_test[:, 0] - 0.035
        X_test[:, 1] = (X_test[:, 1] - 0.17) * 1.75

        return X, y, X_test, y_test