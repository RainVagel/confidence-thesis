from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Layer, Add
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input, AveragePooling2D, LeakyReLU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import TruncatedNormal, Constant


class CustomHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        #self.epoch.append(epoch)
        weights = self.model.layers[-1].get_weights()
        #for k, v in logs.items():
        #    self.history.setdefault(k, []).append(v)
        #self.history.setdefault('c', []).append(weights[0].tolist())
        #self.history.setdefault('b', []).append(weights[1].tolist())
        logs.setdefault('c', []).extend(weights[0].tolist())
        logs.setdefault('b', []).extend(weights[1].tolist())


class RadialSoftmax(Layer):

    def __init__(self, c_initializer=initializers.Zeros(), b_initializer=initializers.Zeros(),
                 c_trainable=True, b_trainable=True, **kwargs):
        super(RadialSoftmax, self).__init__(**kwargs)
        self.c_initializer = c_initializer
        self.b_initializer = b_initializer
        self.supports_masking = True
        self.c_trainable = c_trainable
        self.b_trainable = b_trainable

    def build(self, input_shape):
        self.c = self.add_weight(name="c",
                                 shape=(input_shape[1],),
                                 initializer=self.c_initializer,
                                 trainable=self.c_trainable)
        self.b = self.add_weight(name="b",
                                 shape=(input_shape[1],),
                                 initializer=self.b_initializer,
                                 trainable=self.b_trainable)
        super(RadialSoftmax, self).build(input_shape)

    def call(self, inputs):
        first_exp = tf.exp(self.c - tf.abs(inputs))

        p = (first_exp + tf.exp(self.b)) / tf.reduce_sum(first_exp + tf.exp(self.b), axis=1, keepdims=True)

        return p

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'c_initializer': self.c_initializer, 'b_initializer': self.b_initializer,
                'c_trainable': self.c_trainable, 'b_trainable': self.b_trainable}


class BasicModel:

    """
    # Based on work by Hein, M. et al. from https://github.com/max-andr/relu_networks_overconfident/blob/master/models.py
    """

    def __init__(self, file_name=None, run_name=None, mact=True):
        self.mact = mact
        self.file_name = file_name
        self.run_name = run_name

    def save_model(self, model, folder_name=None, file_name=None):
        if folder_name is None:
            folder_name = self.file_name
        if file_name is None:
            file_name = self.run_name
        model.save(folder_name + "/" + file_name + '.h5')

    def _batch_norm(self, X):
        X = BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True)(X)
        return X

    def _dropout(self, X):
        X = Dropout(0.5)(X)
        return X

    def _global_avg_pool(self, X):
        assert X.get_shape().ndims == 4
        return tf.reduce_mean(X, [1, 2])

    def _residual(self, X, in_filter, out_filter, stride, activate_before_residual=False):
        if activate_before_residual:
            X = self._batch_norm(X)
            X = Activation('relu')(X)
            orig_X = X
        else:
            orig_X = X
            X = self._batch_norm(X)
            X = Activation('relu')(X)

        # Sub1
        X = self._conv(X, filter_size=3, out_filters=out_filter, stride=stride)

        # Sub2
        X = self._batch_norm(X)
        X = Activation('relu')(X)
        X = self._conv(X, filter_size=3, out_filters=out_filter, stride=1)

        #Sub Add
        if in_filter != out_filter:
            orig_X = AveragePooling2D(pool_size=(stride, stride), strides=(stride, stride), padding='valid')(orig_X)
            orig_X = tf.pad(orig_X, [[0, 0], [0, 0], [0, 0],
                                     [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        X = Add()([X, orig_X])
        return X

    def _conv(self, X, filter_size, stride, out_filters, biases=False):
        if biases:
            X = Conv2D(filters=out_filters, kernel_size=(filter_size, filter_size),
                       strides=[stride, stride], padding='same', bias_initializer=Constant(0.0),
                       kernel_regularizer=regularizers.l2(0.0005),
                       bias_regularizer=regularizers.l2(0.0005))(X)
        else:
            X = Conv2D(filters=out_filters, kernel_size=(filter_size, filter_size),
                       strides=[stride, stride], padding='same', kernel_regularizer=regularizers.l2(0.0005),
                       bias_regularizer=regularizers.l2(0.0005))(X)
        return X

    def _fc_layer(self, X, n_out, bn=False, last=False, activation='relu'):
        if len(X.shape) == 4:
            n_in = int(X.shape[1]) * int(X.shape[2]) * int(X.shape[3])
            X = Flatten()(X)
        else:
            n_in = int(X.shape[1])
        X = Dense(n_out, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n_in)),
                  bias_initializer=Constant(0.0), kernel_regularizer=regularizers.l2(0.0005),
                  bias_regularizer=regularizers.l2(0.0005))(X)
        X = self._batch_norm(X) if bn else X
        if not last:
            if activation == 'leakyrelu':
                X = LeakyReLU(alpha=0.3)(X)
            else:
                X = Activation(activation)(X)
        else:
            if self.mact:
                X = RadialSoftmax()(X)
            else:
                X = Activation('softmax')(X)
        return X

    def _conv_layer(self, X, size, n_out, stride, bn=False, biases=True):
        X = self._conv(X, size, stride, n_out, biases=biases)
        X = self._batch_norm(X) if bn else X
        X = Activation('relu')(X)
        return X


class ResNetSmallRunner(BasicModel):

    """
    Based on implementation by Hein, M. et al. on the ResNetSmall model by from https://github.com/max-andr/relu_networks_overconfident/blob/master/models.py
    """

    def __init__(self, mact):
        super().__init__(mact=mact)
        self.n_filters = [16, 16, 32, 64]

    def load_model(self, input_shape, num_classes):
        strides = [1, 1, 2, 2]
        activate_before_residual = [True, False, False]
        n_resid_units = [0, 3, 3, 3]

        X_input = Input(input_shape)
        X = self._conv(X_input, filter_size=3, out_filters=self.n_filters[0],
                       stride=strides[0])
        for i in range(1, len(n_resid_units)):
            X = self._residual(X, self.n_filters[i-1], self.n_filters[i], strides[i], activate_before_residual[0])
            for j in range(1, n_resid_units[i]):
                X = self._residual(X, self.n_filters[i], self.n_filters[i], 1, False)

        # Unit Last
        X = self._batch_norm(X)
        X = Activation('relu')(X)
        X = self._global_avg_pool(X)

        # Logit
        X = self._fc_layer(X, num_classes, last=True)

        model = Model(inputs=X_input, outputs=X, name="ResNetSmall")
        return model


class LeNetRunner(BasicModel):

    """
    Based on implementation by Hein, M. et al. on the LeNet model
    from here: https://github.com/max-andr/relu_networks_overconfident/blob/master/models.py
    """

    def __init__(self, mact, activation='relu'):
        super().__init__(mact=mact)
        self.strides = [1, 1]
        self.n_filters = [32, 64]
        self.n_fc = [1024]
        self.activation = activation

    def load_model(self, input_shape, num_classes):
        bn = False
        X_input = Input(input_shape)
        X = self._conv_layer(X_input, 5, self.n_filters[0], self.strides[0], bn=bn, biases=not bn)
        X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X)
        X = self._conv_layer(X, 5, self.n_filters[1], self.strides[1], bn=bn, biases=not bn)
        X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X)
        X = self._fc_layer(X, self.n_fc[0], activation=self.activation)
        X = self._fc_layer(X, num_classes, last=True)

        model = Model(inputs=X_input, outputs=X, name="LeNet")
        return model
