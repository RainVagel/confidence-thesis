from __future__ import print_function

import csv

import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import Constant
from tensorflow.keras.initializers import TruncatedNormal

from analysis import BaseAnalyser
from sklearn.utils import shuffle


class CustomHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        weights = self.model.layers[-1].get_weights()
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.history.setdefault('c', []).append(weights[0].tolist())
        self.history.setdefault('b', []).append(weights[1].tolist())


class MAct(Layer):
    """
    Original modified softmax function.
    """

    def __init__(self, c_initializer=initializers.Ones(), b_initializer=initializers.Ones(),
                 c_trainable=True, b_trainable=True, **kwargs):
        super(MAct, self).__init__(**kwargs)
        self.c_initializer = c_initializer
        self.b_initializer = b_initializer
        self.supports_masking = True
        self.c_trainable = c_trainable
        self.b_trainable = b_trainable

    def build(self, input_shape):
        self.c = self.add_weight(name="c",
                                 shape=(input_shape[1],),
                                 initializer=self.c_initializer,
                                 trainable=self.c_trainable)  # Initialiseerida c ühtedeks / nullideks
        self.b = self.add_weight(name="b",
                                 shape=(input_shape[1],),
                                 initializer=self.b_initializer,
                                 trainable=self.b_trainable)  # Initialiseerida b nullideks
        super(MAct, self).build(input_shape)

    def call(self, inputs):
        first_exp = tf.exp(self.c - tf.square(inputs))

        p = (first_exp + tf.exp(self.b)) / tf.reduce_sum(first_exp + tf.exp(self.b), axis=1, keepdims=True)

        # p = tf.exp(inputs) / tf.reduce_sum(tf.exp(inputs), axis=0, keepdims=True)
        return p

    def compute_output_shape(self, input_shape):
        return input_shape


class MActAbs(MAct):
    """
    Newer modified softmax that we tried during the meeting at Delta held on the 16th of January.

    Instead of squaring the inputs we take the absolute value and b is negative.
    """

    def call(self, inputs):
        first_exp = tf.exp(self.c - tf.abs(inputs))

        p = (first_exp + tf.exp(-self.b)) / tf.reduce_sum(first_exp + tf.exp(-self.b), axis=1, keepdims=True)

        # p = tf.exp(inputs) / tf.reduce_sum(tf.exp(inputs), axis=0, keepdims=True)
        return p


class ModelRunner:
    def __init__(self, model, file_name, run_name, iterations, dim):
        self.model = model
        self.file_name = file_name
        self.run_name = run_name
        self.iterations = iterations
        self.dim = dim

    def get_model(self):
        return self.model

    def cross_ent(self, logits, y):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)  # Tavaline CE
        return tf.reduce_mean(losses)

    def max_conf(self, logits):
        y = tf.argmax(logits, 1)
        y = tf.one_hot(y, self.dim)
        losses = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)  # Tavaline CE
        return tf.reduce_mean(losses)

    def gen_adv(self, x):
        eps = 0.025
        n_iters = 4
        step_size = 0.02

        unif = tf.random.uniform(minval=-eps, maxval=eps, shape=tf.shape(x))
        x_adv = tf.clip_by_value(x + unif, 0., 1.)

        for i in range(n_iters):
            x_adv = tf.Variable(x_adv)
            with tf.GradientTape() as tape:
                loss = self.max_conf(self.model(x_adv))
                grad = tape.gradient(loss, x_adv)
                g = tf.sign(grad)

            x_adv_start = x_adv + step_size * g
            x_adv = tf.clip_by_value(x_adv, 0., 1.)
            delta = x_adv - x_adv_start
            delta = tf.clip_by_value(delta, -eps, eps)
            x_adv = x_adv_start + delta

        return x_adv

    def experiment_logging(self, logits, X, y, train_err, epoch, loss_main, loss_acet, loss_curve,
                           iter_list, info_list, inter_plots, analyser):
        print("Iter {:03d}: loss_main={:.10f} loss_acet={:.3f} err={:.2%}"
              .format(epoch, loss_main, loss_acet, train_err))

        loss_curve.append(loss_main)
        iter_list.append(epoch)

        if inter_plots:
            analyser.plot(self.model, 0.0, 2.0, True, self.file_name, self.run_name + "_inter_plot_epoch={}"
                          .format(epoch), X, y)
            analyser.plot(self.model, -10.0, 10.0, True, self.file_name, self.run_name + "_inter_plot_epoch={}"
                          .format(epoch), X, y)
            analyser.get_output(model=self.model, X=X, layer=-2, file_name=self.file_name,
                                layers=self.run_name + "_inter_plot_epoch={}".format(epoch))

        info_list.append("Iter {:03d}: loss_main={:.10f} loss_acet={:.6f} err={:.2%}"
                         .format(epoch, loss_main, loss_acet, train_err))
        return loss_curve, iter_list, info_list

    def model_experiment(self, optimizer, acet, X, y, X_test, y_test, batch_size=0,
                         buffer_size=10000, save_step=100, inter_plots=False):
        print("Model experiment starting")
        analyser = BaseAnalyser()
        info_list = []
        loss_curve = []
        iter_list = []

        # Custom training cycle going through the entire dataset
        for epoch in range(1, self.iterations + 1):
            errors = []
            if batch_size != 0:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
                train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
                for step, (x_batch, y_batch) in enumerate(train_dataset):
                    X_noise = tf.random.uniform([2 * x_batch.shape[0], x_batch.shape[1]])
                    # If we use the ACET method, then adversarial noise will be generated
                    if acet:
                        X_noise = self.gen_adv(X_noise)
                    # Context used to calculate the gradients of the model
                    with tf.GradientTape() as tape:
                        logits = self.model(x_batch)
                        logits_noise = self.model(X_noise)
                        loss_main = self.cross_ent(logits, y_batch)
                        loss_acet = acet * self.max_conf(logits_noise)
                        loss = loss_main + loss_acet
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    errors.append(np.mean(logits.numpy().argmax(1) != y_batch.numpy().argmax(1)))
                if epoch % save_step == 0:
                    print(epoch)
                    loss_curve, iter_list, info_list = self.experiment_logging(
                        logits, X, y, epoch, np.mean(errors), loss_main, loss_acet, loss_curve, iter_list,
                        info_list, inter_plots, analyser)
            else:
                X_noise = tf.random.uniform([2 * x_batch.shape[0], x_batch.shape[1]])
                # If we use the ACET method, then adversarial noise will be generated
                if acet:
                    X_noise = self.gen_adv(X_noise)
                # Context used to calculate the gradients of the model
                with tf.GradientTape() as tape:
                    logits = self.model(X)
                    logits_noise = self.model(X_noise)
                    loss_main = self.cross_ent(logits, y)
                    loss_acet = acet * self.max_conf(logits_noise)
                    loss = loss_main + loss_acet
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                train_err = np.mean(logits.numpy().argmax(1) != y.numpy().argmax(1))
                if epoch % save_step == 0:
                    print(epoch)
                    loss_curve, iter_list, info_list = self.experiment_logging(
                        logits, X, y, epoch, train_err, loss_main, loss_acet, loss_curve, iter_list,
                        info_list, inter_plots, analyser)

        file_name = "{}/{}_iters={}.csv".format(self.file_name, self.run_name, self.iterations)

        print("Starting plotting!")
        analyser.write_log(file_name, info_list)
        analyser.plot(self.model, 0.0, 1.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(self.model, -2.0, 3.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(self.model, -5.0, 6.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(self.model, -10.0, 10.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.single_output_plot(model=self.model, plot_min=0.0, plot_max=2.0, layer=-2,
                                    file_name=self.file_name, layers=self.run_name + "_epoch={}"
                                    .format(self.iterations))
        print("Plotting completed")

    def save_model(self, folder_name=None, file_name=None):
        if folder_name is None:
            folder_name = self.file_name
        if file_name is None:
            file_name = self.run_name
        self.model.save(folder_name + "/" + file_name)

    def save_layer_weights(self, folder_name=None, file_name=None, layer=-1):
        if folder_name is None:
            folder_name = self.file_name
        if file_name is None:
            file_name = self.run_name
        file_name = "{}/{}.csv".format(folder_name, file_name)
        analyser = BaseAnalyser()

        if layer == 'all':
            info_list = []
            for layer in self.model.layers:
                info_list.append(layer.get_weights())
            analyser.write_log(file_name, info_list)
        else:
            analyser.write_log(file_name, self.model.layers[layer].get_weights())

    def predict(self, data):
        return self.model.predict(data)

    def get_predict_confidence(self, X):
        predictions = self.model.predict(X)
        preds_conf = []
        for prediction in predictions:
            preds_conf.append((prediction.argmax(-1), prediction[prediction.argmax(-1)]))
        return preds_conf

    def evaluate(self, X, y):
        accuracy_metric = CategoricalAccuracy()
        return accuracy_metric(y, self.model(X))


class MActModelRunner(ModelRunner):
    def cross_ent(self, probs, y):
        cce = CategoricalCrossentropy()
        losses = cce(probs, y)
        return tf.reduce_mean(losses)

    def max_conf(self, probs):
        y = tf.argmax(probs, 1)
        y = tf.one_hot(y, self.dim)
        cce = CategoricalCrossentropy()
        losses = -cce(probs, y)
        return tf.reduce_mean(losses)

    def experiment_logging(self, logits, X, y, epoch, train_err, loss_main, loss_acet, loss_curve,
                           iter_list, info_list, inter_plots, analyser):
        print("Iter {:03d}: loss_main={:.10f} loss_acet={:.3f} err={:.2%}"
              .format(epoch, loss_main, loss_acet, train_err))

        loss_curve.append(loss_main)
        iter_list.append(epoch)

        if inter_plots:
            analyser.plot(self.model, 0.0, 2.0, True, self.file_name, self.run_name + "_inter_plot_epoch={}"
                          .format(epoch), X, y)
            analyser.plot(self.model, -10.0, 10.0, True, self.file_name, self.run_name + "_inter_plot_epoch={}"
                          .format(epoch), X, y)
            analyser.single_output_plot(model=self.model, plot_min=0.0, plot_max=2.0, layer=-2,
                                        file_name=self.file_name, layers=self.run_name + "_inter_plot_epoch={}"
                                        .format(epoch))

        weights = self.model.layers[-1].get_weights()
        info_list.append("Iter {:03d}: loss_main={:.10f} loss_acet={:.6f} err={:.2%} c: {}, b: {}"
                         .format(epoch, loss_main, loss_acet, train_err, weights[0], weights[1]))
        return loss_curve, iter_list, info_list


class CifarModelRunner(ModelRunner):

    def __init__(self, model, file_name, run_name, iterations, dim, data_augmentation=False):
        super().__init__(model, file_name, run_name, iterations, dim)
        self.data_augmentation = data_augmentation

    def load_keras_cifar10_model(self, x_train, num_classes=10, mact=False):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              input_shape=x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        if mact:
            self.model.add(MActAbs())
        else:
            self.model.add(Activation('softmax'))

    def get_default_optimizer(self):
        return RMSprop(learning_rate=0.0001, decay=1e-6)

    def compile_model(self, opt, loss='categorical_crossentropy', metrics='accuracy'):
        self.model.compile(loss=loss,
                           optimizer=opt,
                           metrics=[metrics])

    def get_history(self):
        return self.history

    def model_experiment(self, optimizer, X, y, X_test, y_test, batch_size=0, shuffle=True,
                         buffer_size=10000, save_step=100, workers=0):
        self.history = CustomHistory()
        if not self.data_augmentation:
            self.model.fit(X, y, batch_size=batch_size, epochs=self.iterations,
                           validation_data=(X_test, y_test), shuffle=shuffle, callbacks=[self.history])
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(X)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size), epochs=self.iterations,
                                                    validation_data=(X_test, y_test), workers=workers, callbacks=[self.history])


class LeNetRunner(ModelRunner):

    # Based on LeNet from here: https://github.com/MadryLab/mnist_challenge/blob/master/model.py

    def __init__(self, model, file_name, run_name, iterations, dim, mact):
        super().__init__(model, file_name, run_name, iterations, dim)
        self.mact = mact

    def load_model(self, x_train, num_classes):
        self.model = Sequential()
        # First conv layer
        self.model.add(Conv2D(32, kernel_size=(5, 5), kernel_initializer=TruncatedNormal(stddev=0.1),
                              bias_initializer=Constant(value=0.1), strides=[1, 1, 1, 1], padding='same',
                              input_size=x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second conv layer
        self.model.add(Conv2D(64, kernel_size=(5, 5), kernel_initializer=TruncatedNormal(stddev=0.1),
                              bias_initializer=Constant(value=0.1), strides=[1, 1, 1, 1], padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # First Dense layer
        self.model.add(Dense(1024, kernel_initializer=TruncatedNormal(stddev=0.1),
                             bias_initializer=Constant(value=0.1)))
        self.model.add(Flatten())
        self.model.add(Activation('relu'))

        #Output layer
        self.model.add(Dense(num_classes, kernel_initializer=TruncatedNormal(stddev=0.1),
                             bias_initializer=Constant(value=0.1)))
        if self.mact:
            self.model.add(MActAbs())
        else:
            self.model.add(Activation('softmax'))

    def compile_model(self, opt, loss='sparse_categorical_crossentropy', metrics='accuracy'):
        self.model.compile(loss=loss,
                           optimizer=opt,
                           metrics=[metrics])

    def get_default_optimizer(self):
        return Adam(learning_rate=1e-4)

