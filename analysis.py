import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras import backend as K


class BaseAnalyser:
    def __init__(self, m_act=True):
        self.m_act = m_act

    def get_output(self, X, model, layer):
        layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
        output = layer_output([X])[0]
        return output

    def single_output_plot(self, model, layer, file_name, layers, plot_min, plot_max):
        n_grid = 200
        x_plot = np.linspace(plot_min, plot_max, n_grid)
        y_plot = np.linspace(plot_min, plot_max, n_grid)

        points = []
        for xx in x_plot:
            for yy in y_plot:
                points.append((yy, xx))
        points = np.array(points)

        output = self.get_output(points, model, layer)

        column_nr = np.size(output, 1)

        for column in range(0, column_nr):
            z_plot = output[:, column]
            z_plot = z_plot.reshape(len(x_plot), len(y_plot)) * 100

            vmax = np.max(z_plot)
            vmin = np.min(z_plot)
            plt.subplot(1, column_nr, column+1)
            plt.contourf(x_plot, y_plot, z_plot, levels=np.linspace(vmin, vmax, 5), cmap='seismic')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Z_{}".format(column))

        plt.savefig('{}/{}_{:.1f}_{:.1f}.pdf'.format(
            file_name, layers, plot_min, plot_max), transparent=True)
        plt.clf()

    def plot(self, model, plot_min, plot_max, max_prob, file_name, layers, X, y):
        n_grid = 200
        x_plot = np.linspace(plot_min, plot_max, n_grid)
        y_plot = np.linspace(plot_min, plot_max, n_grid)

        points = []
        for xx in x_plot:
            for yy in y_plot:
                points.append((yy, xx))
        points = np.array(points)

        if self.m_act:
            probs = model(points).numpy()
        else:
            logits = model(points)
            probs = tf.nn.softmax(logits).numpy()

        if max_prob:
            z_plot = probs.max(1)
        else:
            z_plot = probs[:, 0]
        z_plot = z_plot.reshape(len(x_plot), len(y_plot)) * 100

        vmax = 100
        vmin = 50 if max_prob else 0
        plt.contourf(x_plot, y_plot, z_plot, levels=np.linspace(50, 100, 50))
        cbar = plt.colorbar(ticks=np.linspace(vmin, vmax, 6))

        cbar.ax.set_title('confidence', fontsize=12, pad=12)
        cbar.set_ticklabels(['50%', '60%', '70%', '80%', '90%', '100%'])

        y_np = np.array(y)
        X0 = X[y_np.argmax(1) == 0]
        X1 = X[y_np.argmax(1) == 1]
        plt.scatter(X0[:, 0], X0[:, 1], s=20, edgecolors='red', facecolor='None',
                    marker='o', linewidths=0.2)
        plt.scatter(X1[:, 0], X1[:, 1], s=20, edgecolors='green', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.xlim([plot_min, plot_max])
        plt.ylim([plot_min, plot_max])

        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('{}/{}_{:.1f}_{:.1f}_max_prob={}.pdf'.format(
            file_name, layers, plot_min, plot_max, max_prob), transparent=True)
        plt.clf()

    def write_log(self, file_name, log):
        with open(file_name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter="\n")
            wr.writerow(log)

    def generate_loss_curve(self, file_name, loss_list, iter_list):
        plt.plot(iter_list, loss_list)
        plt.xticks(iter_list, iter_list)
        plt.title("Loss curve")
        plt.savefig('{}_loss_curve.png'.format(file_name))
        plt.clf()
