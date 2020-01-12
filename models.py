from __future__ import print_function

import csv

import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Layer

from analysis import BaseAnalyser


class MAct(Layer):

    def __init__(self, **kwargs):
        super(MAct, self).__init__(**kwargs)
        self.c_initializer = initializers.Ones()
        self.b_initializer = initializers.Ones()
        self.supports_masking = True

    def build(self, input_shape):
        self.c = self.add_weight(name="c",
                                 shape=(input_shape[1],),
                                 initializer=self.c_initializer,
                                 trainable=True)  # Initialiseerida c ühtedeks / nullideks
        self.b = self.add_weight(name="b",
                                 shape=(input_shape[1],),
                                 initializer=self.b_initializer,
                                 trainable=True)  # Initialiseerida b nullideks
        super(MAct, self).build(input_shape)

    def call(self, inputs):
        first_exp = tf.exp(self.c - tf.square(inputs))

        p = (first_exp + tf.exp(self.b)) / tf.reduce_sum(first_exp + tf.exp(self.b), axis=1, keepdims=True)

        # p = tf.exp(inputs) / tf.reduce_sum(tf.exp(inputs), axis=0, keepdims=True)
        return p

    def get_config(self):
        config = {
            'c_initializer': initializers.serialize(self.c_initializer),
            'b_initializer': initializers.serialize(self.b_initializer),
        }
        base_config = super(MAct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ModelRunner:
    def __init__(self, file_name, run_name, iterations, dim):
        self.file_name = file_name
        self.run_name = run_name
        self.iterations = iterations
        self.dim = dim

    def cross_ent(self, logits, y):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)  # Tavaline CE
        return tf.reduce_mean(losses)

    def max_conf(self, logits):
        y = tf.argmax(logits, 1)
        y = tf.one_hot(y, self.dim)
        losses = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)  # Tavaline CE
        return tf.reduce_mean(losses)

    def gen_adv(self, model, x):
        eps = 0.025
        n_iters = 4
        step_size = 0.02

        unif = tf.random.uniform(minval=-eps, maxval=eps, shape=tf.shape(x))
        x_adv = tf.clip_by_value(x + unif, 0., 1.)

        for i in range(n_iters):
            x_adv = tf.Variable(x_adv)
            with tf.GradientTape() as tape:
                loss = self.max_conf(model(x_adv))
                grad = tape.gradient(loss, x_adv)
                g = tf.sign(grad)

            x_adv_start = x_adv + step_size * g
            x_adv = tf.clip_by_value(x_adv, 0., 1.)
            delta = x_adv - x_adv_start
            delta = tf.clip_by_value(delta, -eps, eps)
            x_adv = x_adv_start + delta

        return x_adv

    def model_experiment(self, model, optimizer, acet, X, y, X_test, y_test):
        print("Model experiment starting")
        info_list = []

        # Custom training cycle going through the entire dataset
        for epoch in range(1, self.iterations + 1):
            X_noise = tf.random.uniform([2 * X.shape[0], X.shape[1]])
            # If we use the ACET method, then adversarial noise will be generated
            if acet:
                X_noise = self.gen_adv(X_noise, model)
            # Context used to calculate the gradients of the model
            with tf.GradientTape() as tape:
                logits = model(X)
                logits_noise = model(X_noise)
                loss_main = self.cross_ent(logits, y)
                loss_acet = acet * self.max_conf(logits_noise)
                loss = loss_main + loss_acet
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % 100 == 0:
                train_err = np.mean(logits.numpy().argmax(1) != y.numpy().argmax(1))
                print("Iter {:03d}: loss_main={:.10f} loss_acet={:.3f} err={:.2%}"
                      .format(epoch, loss_main, loss_acet, train_err))

                info_list.append("Iter {:03d}: loss_main={:.10f} loss_acet={:.6f} err={:.2%}"
                                 .format(epoch, loss_main, loss_acet, train_err))

        file_name = "{}/{}_iters={}.csv".format(self.file_name, self.run_name, self.iterations)
        with open(file_name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter="\n")
            wr.writerow(info_list)

        print("Starting plotting!")
        analyser = BaseAnalyser()
        analyser.plot(model, 0.0, 1.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -2.0, 3.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -5.0, 6.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -10.0, 10.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        print("Plotting completed")


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

    def model_experiment(self, model, optimizer, acet, X, y, X_test, y_test):
        print("Model experiment starting")
        info_list = []

        # Custom training cycle going through the entire dataset
        for epoch in range(1, self.iterations + 1):
            X_noise = tf.random.uniform([2 * X.shape[0], X.shape[1]])
            # If we use the ACET method, then adversarial noise will be generated
            if acet:
                X_noise = self.gen_adv(X_noise, model)
            # Context used to calculate the gradients of the model
            with tf.GradientTape() as tape:
                logits = model(X)
                logits_noise = model(X_noise)
                loss_main = self.cross_ent(logits, y)
                loss_acet = acet * self.max_conf(logits_noise)
                loss = loss_main + loss_acet
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % 100 == 0:
                train_err = np.mean(logits.numpy().argmax(1) != y.numpy().argmax(1))
                print("Iter {:03d}: loss_main={:.10f} loss_acet={:.3f} err={:.2%}"
                      .format(epoch, loss_main, loss_acet, train_err))

                weights = model.layers[-1].get_weights()
                info_list.append("Iter {:03d}: loss_main={:.10f} loss_acet={:.6f} err={:.2%} c: {}, b: {}"
                                 .format(epoch, loss_main, loss_acet, train_err, weights[0], weights[1]))

        file_name = "{}/{}_iters={}.csv".format(self.file_name, self.run_name, self.iterations)
        with open(file_name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter="\n")
            wr.writerow(info_list)

        print("Starting plotting!")
        analyser = BaseAnalyser()
        analyser.plot(model, 0.0, 1.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -2.0, 3.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -5.0, 6.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        analyser.plot(model, -10.0, 10.0, True, self.file_name, self.run_name + "_iters={}"
                      .format(self.iterations), X, y)
        print("Plotting completed")

