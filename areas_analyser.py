from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf

from models import RadialSoftmax
from analysis import BaseAnalyser
from dataset import MoonsDataset


def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d


def dataset_generator(x_start, x_end, y_start, y_end, step):
    X = []
    # Upper edge
    for x in np.arange(x_start, x_end, step):
        x = round(x, 2)
        X.append([x, y_end-0.01])

    # Right side
    for y in np.arange(y_end, y_start, -step):
        y = round(y, 2)
        X.append([x_end-0.01, y])

    # Lower edge
    for x in np.arange(x_end, x_start, -step):
        x = round(x, 2)
        X.append([x, y_start])

    # Left side
    for y in np.arange(y_start, y_end, step):
        y = round(y, 2)
        X.append([x_start, y])

    return X, tf.convert_to_tensor(X)


def model_load(model):
    try:
        loaded_model = load_model(model, custom_objects={'RadialSoftmax': RadialSoftmax})
    except ValueError:
        loaded_model = load_model(model, custom_objects={'RadialSoftmax': RadialSoftmax})
    return loaded_model

def all_model_analysis(folder):
    files_list = os.walk(folder)
    analyser = BaseAnalyser()
    #results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    #print(files_list)
    for stuff in files_list:
        print(stuff)
        if '/' in stuff[0]:
            name = stuff[0]
            model = name + '/model.h5'
            loaded_model = model_load(model)

            X, tensor_X = dataset_generator(-1, 3, -1, 3, 0.25)
            output = analyser.get_output(tensor_X, loaded_model, -2)

            z1_arr = [z_value[0] for z_value in output]
            z2_arr = [z_value[1] for z_value in output]

            plt.plot(z1_arr, label='z1')
            plt.plot(z2_arr, label='z2')
            plt.legend()
            plt.xticks([])
            plt.title("Z1 and Z2 values Square from -1 to 3")
            plt.savefig(name + '/z1_z2_experiment.png')
            plt.clf()


def radial_softmax(inputs, c, b):
    first_exp = np.exp(c - np.abs(inputs))

    p = (first_exp + np.exp(b)) / np.sum(first_exp + np.exp(b), axis=1, keepdims=True)

    return p

def radial_softmax_temp(inputs, c, b, temp):
    first_exp = np.exp(c - np.abs(inputs/temp))

    p = (first_exp + np.exp(b)) / np.sum(first_exp + np.exp(b), axis=1, keepdims=True)

    return p


def folder_creater(file_name):
    Path(file_name).mkdir(parents=True, exist_ok=True)

def cross_ent(probs, y):
    #losses = tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=y) # Tavaline CE
    cce = CategoricalCrossentropy()
    losses = cce(probs, y)
    return tf.reduce_mean(losses)


def softmax_play_plot(model, plot_min, plot_max, max_prob, file_name, layers, X, y, c, b, classes=2, temp=None):
    """
    Based on Hein, M. et al. :https://github.com/max-andr/relu_networks_overconfident/blob/master/analysis.py
    """
    n_grid = 200
    x_plot = np.linspace(plot_min, plot_max, n_grid)
    y_plot = np.linspace(plot_min, plot_max, n_grid)

    points = []
    for xx in x_plot:
        for yy in y_plot:
            points.append((yy, xx))
    points = np.array(points)

    analyser = BaseAnalyser()

    probs = analyser.get_output(points, model, -2)
    x_probs = analyser.get_output(X, model, -2)
    if temp is None:
        probs = radial_softmax(probs, c, b)
        x_probs = radial_softmax(x_probs, c, b)
    else:
        probs = radial_softmax_temp(probs, c, b, temp)
        x_probs = radial_softmax_temp(x_probs, c, b, temp)
    z_plot = probs.max(1)

    z_plot = z_plot.reshape(len(x_plot), len(y_plot)) * 100

    loss = np.round(np.asarray(cross_ent(x_probs, y)), 4)
    mmc = np.round(np.asarray(np.mean(analyser.max_conf(x_probs))), 4)


    layers = layers + '_mmc={}_loss={}'.format(mmc, loss)

    vmax = 100
    vmin = 50 if max_prob else 0
    if classes == 8:
        vmin = 10
        plt.contourf(x_plot, y_plot, z_plot, levels=np.linspace(10, 100, 90))
        cbar = plt.colorbar(ticks=np.linspace(vmin, vmax, 10))

        cbar.ax.set_title('confidence', fontsize=12, pad=12)
        cbar.set_ticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    else:
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
    if classes == 8:
        X2 = X[y_np.argmax(1) == 2]
        X3 = X[y_np.argmax(1) == 3]
        X4 = X[y_np.argmax(1) == 4]
        X5 = X[y_np.argmax(1) == 5]
        X6 = X[y_np.argmax(1) == 6]
        X7 = X[y_np.argmax(1) == 7]
        plt.scatter(X2[:, 0], X2[:, 1], s=20, edgecolors='black', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X3[:, 0], X3[:, 1], s=20, edgecolors='gray', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X4[:, 0], X4[:, 1], s=20, edgecolors='m', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X5[:, 0], X5[:, 1], s=20, edgecolors='darksalmon', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X6[:, 0], X6[:, 1], s=20, edgecolors='tan', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X7[:, 0], X7[:, 1], s=20, edgecolors='olivedrab', facecolor='None',
                    marker='s', linewidths=0.2)
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('{}/{}_{:.1f}_{:.1f}_max_prob={}.pdf'.format(
        file_name, layers, plot_min, plot_max, max_prob), transparent=True)
    plt.clf()


def softmax_layer_play(model_folder, pos=True):
    loaded_model = model_load(model_folder + '/model.h5')
    analyser = BaseAnalyser()
    moons = MoonsDataset()

    if pos:
        fixed_b = [1, 1]
    else:
        fixed_b = [-1, -1]

    X, y, X_test, y_test = moons.four_set_two_moons()
    #print(X)

    if pos:
        output_folder = model_folder + '/softmax_modding_pos'
    else:
        output_folder = model_folder + '/softmax_modding_minus'
    folder_creater(output_folder)

    c_values = []
    for c_1 in range(1, 11, 2):
        for c_2 in range(1, 11, 2):
            c_values.append([c_1, c_2])

    #print(model(X))
    #print(radial_softmax(model(X), [5, 5], [1, 1]))

    for c in c_values:
        softmax_play_plot(loaded_model, -2.0, 3.0, True, output_folder, 'c={}_b={}'.format(c, fixed_b),
                          X, y, c, fixed_b)
        softmax_play_plot(loaded_model, 0.0, 2.0, True, output_folder,
                          'c={}_b={}'.format(c, fixed_b),
                          X, y, c, fixed_b)


def temperature_scaling(model_folder):
    loaded_model = model_load(model_folder + '/model.h5')
    analyser = BaseAnalyser()
    moons = MoonsDataset()
    fixed_b = [1, 1]
    X, y, X_test, y_test = moons.four_set_two_moons()
    # print(X)

    output_folder = model_folder + '/temperature_scaling'
    folder_creater(output_folder)

    #print(loaded_model.layers[-1].weights)
    c = loaded_model.layers[-1].get_weights()[0]
    b = loaded_model.layers[-1].get_weights()[1]
    #print(loaded_model.layers[-1].get_weights())

    for temp in np.arange(0.25, 2.25, 0.25):
        softmax_play_plot(loaded_model, -2.0, 3.0, True, output_folder,
                          'c={}_b={}_temp={}'.format(c, b, temp),
                          X, y, c, b, temp=temp)
        softmax_play_plot(loaded_model, 0.0, 2.0, True, output_folder,
                          'c={}_b={}_temp={}'.format(c, b, temp),
                          X, y, c, b, temp=temp)
        softmax_play_plot(loaded_model, -5.0, 6.0, True, output_folder,
                          'c={}_b={}_temp={}'.format(c, b, temp),
                          X, y, c, b, temp=temp)

def get_ticks(X, step):
    ticks = X[0::int(len(X) / step)]
    return [x[0] for x in ticks], [X.index(x) for x in ticks]

def softmax_problems_large_scores(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    # (0.5, 3.25, 0.25)
    for x_coord in np.arange(1, 4.01, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -2)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]

    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    x_ticks, x_ticks_pos = get_ticks(X, 5)
    plt.legend()
    #plt.xticks([0, 3, 7, 11], ['0.5', '1.25', '2.25', '3.25'])
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xticks(x_ticks_pos, x_ticks)
    plt.xlabel('x')
    plt.ylabel('Class Scores')
    #plt.show()
    plt.savefig(name + '/z_values_experiment_outward.png')
    plt.clf()

def softmax_problems_large_probs(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    # (0.5, 3.25, 0.25)
    for x_coord in np.arange(1, 4.01, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -1)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]

    x_ticks, x_ticks_pos = get_ticks(X, 5)
    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    plt.legend()
    #plt.xticks([0, 3, 7, 11], ['0.5', '1.25', '2.25', '3.25'])
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xticks(x_ticks_pos, x_ticks)
    plt.xlabel('x')
    plt.ylabel('Class Probabilities')
    #plt.show()
    plt.savefig(name + '/probabilities_experiment_outward.png')
    plt.clf()

def softmax_problems_small_scores(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    # (0.5, 3.25, 0.25)
    for x_coord in np.arange(0.5, 3.76, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -2)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]

    x_ticks, x_ticks_pos = get_ticks(X, 5)

    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    plt.legend()
    plt.xticks(x_ticks_pos,x_ticks)
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xlabel('x')
    plt.ylabel('Class Scores')
    #plt.show()
    plt.savefig(name + '/z_values_experiment_outward.png')
    plt.clf()

def softmax_problems_small_probs(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    # (0.5, 3.25, 0.25)
    for x_coord in np.arange(0.5, 3.76, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -1)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]

    x_ticks, x_ticks_pos = get_ticks(X, 5)

    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    plt.legend()
    #plt.xticks([0, 3, 7, 11], ['0.5', '1.25', '2.25', '3.25'])
    plt.xticks(x_ticks_pos,x_ticks)
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xlabel('x')
    plt.ylabel('Class Probabilities')
    #plt.show()
    plt.savefig(name + '/probabilities_experiment_outward.png')
    plt.clf()


def softmax_problems_eigth_probs(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    for x_coord in np.arange(1, 4.01, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -1)


    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]
    z3_arr = [z_value[2] for z_value in output]
    z4_arr = [z_value[3] for z_value in output]
    z5_arr = [z_value[4] for z_value in output]
    z6_arr = [z_value[5] for z_value in output]
    z7_arr = [z_value[6] for z_value in output]
    z8_arr = [z_value[7] for z_value in output]

    x_ticks, x_ticks_pos = get_ticks(X, 5)

    plt.plot(z1_arr, color='red',        label='Class 1')
    plt.plot(z2_arr, color='green',      label='Class 2')
    plt.plot(z3_arr, color='black',      label='Class 3')
    plt.plot(z4_arr, color='gray',        label='Class 4')
    plt.plot(z5_arr, color='m',           label='Class 5')
    plt.plot(z6_arr, color='darksalmon', label='Class 6')
    plt.plot(z7_arr, color='tan',         label='Class 7')
    plt.plot(z8_arr, color='olivedrab',    label='Class 8')
    plt.legend()
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xticks(x_ticks_pos, x_ticks)
    plt.xlabel('x')
    plt.ylabel('Class Probabilities')
    #plt.show()
    plt.savefig(name + '/probabilities_experiment_outward.png')
    plt.clf()

def softmax_problems_eigth_scores(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X = []
    x_coords = []
    for x_coord in np.arange(1, 4.01, 0.01):
        x_coord = round(x_coord, 2)
        X.append([x_coord, 1])
        x_coords.append(x_coord)

    tensor_X = tf.convert_to_tensor(X)
    output = analyser.get_output(tensor_X, loaded_model, -2)


    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]
    z3_arr = [z_value[2] for z_value in output]
    z4_arr = [z_value[3] for z_value in output]
    z5_arr = [z_value[4] for z_value in output]
    z6_arr = [z_value[5] for z_value in output]
    z7_arr = [z_value[6] for z_value in output]
    z8_arr = [z_value[7] for z_value in output]

    x_ticks, x_ticks_pos = get_ticks(X, 5)

    plt.plot(z1_arr, color='red',      label='Class 1')
    plt.plot(z2_arr, color='green',    label='Class 2')
    plt.plot(z3_arr, color='black',    label='Class 3')
    plt.plot(z4_arr, color='gray',     label='Class 4')
    plt.plot(z5_arr, color='m',        label='Class 5')
    plt.plot(z6_arr, color='darksalmon', label='Class 6')
    plt.plot(z7_arr, color='tan',      label='Class 7')
    plt.plot(z8_arr, color='olivedrab',label='Class 8')
    plt.legend()
    #plt.xticks([0, 3, 7, 11, 14], ['1.0', '1.75', '2.75', '3.75', '4.75'])
    plt.xticks(x_ticks_pos, x_ticks)
    plt.xlabel('x')
    plt.ylabel('Class Scores')
    #plt.show()
    plt.savefig(name + '/z_values_experiment_outward.png')
    plt.clf()

def square_info_two_scores(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X, tensor_X = dataset_generator(-2.5, 3.76, -2.5, 3.76, 0.01)
    output = analyser.get_output(tensor_X, loaded_model, -2)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]


    plt.plot(z1_arr, color='red',   label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Class Scores')

    x_ticks = [[-2.5, 3.75], [3.75, 3.75], [3.75, -2.5], [-2.5, -2.5], [-2.5, 3.75]]
    x_tick_pos = [X.index(tick) for tick in x_ticks]
    x_tick_pos[-1] = X[1:].index([-2.5, 3.75])
    plt.xticks(x_tick_pos, [str((x[0], x[1])) for x in x_ticks])

    plt.savefig(name + '/z_values_square_experiment.png')
    plt.clf()
    #plt.show()

def square_info_two_probs(name):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X, tensor_X = dataset_generator(-2.5, 3.76, -2.5, 3.76, 0.01)
    output = analyser.get_output(tensor_X, loaded_model, -1)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]


    plt.plot(z1_arr, color='red',   label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Class Probabilities')
    x_ticks = [[-2.5, 3.75], [3.75, 3.75], [3.75, -2.5], [-2.5, -2.5], [-2.5, 3.75]]
    x_tick_pos = [X.index(tick) for tick in x_ticks]
    x_tick_pos[-1] = X[1:].index([-2.5, 3.75])
    plt.xticks(x_tick_pos, [str((x[0], x[1])) for x in x_ticks])
    plt.savefig(name + '/probabilities_square_experiment.png')
    plt.clf()
    #plt.show()


def square_info_eight_probs(name, classes=8):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X, tensor_X = dataset_generator(-2, 4.01, -2, 4.01, 0.01)
    output = analyser.get_output(tensor_X, loaded_model, -1)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]
    if classes == 8:
        z3_arr = [z_value[2] for z_value in output]
        z4_arr = [z_value[3] for z_value in output]
        z5_arr = [z_value[4] for z_value in output]
        z6_arr = [z_value[5] for z_value in output]
        z7_arr = [z_value[6] for z_value in output]
        z8_arr = [z_value[7] for z_value in output]

    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    if classes == 8:
        plt.plot(z3_arr, color='black',      label='Class 3')
        plt.plot(z4_arr, color='gray',       label='Class 4')
        plt.plot(z5_arr, color='m',          label='Class 5')
        plt.plot(z6_arr, color='darksalmon', label='Class 6')
        plt.plot(z7_arr, color='tan',        label='Class 7')
        plt.plot(z8_arr, color='olivedrab',  label='Class 8')
    plt.legend()
    x_ticks = [[-2.0, 4.0], [4.0, 4.0], [4.0, -2.0], [-2.0, -2.0], [-2.0, 4.0]]
    x_tick_pos = [X.index(tick) for tick in x_ticks]
    x_tick_pos[-1] = X[1:].index([-2.0, 4.0])
    plt.xticks(x_tick_pos, [str((x[0], x[1])) for x in x_ticks])
    #plt.xticks([0, 25, 50, 75, 99], ['(-2.0, 4.0)', '(4.0, 4.0)', '(4.0, -2.0)', '(-2.0, -2.0)', '(-2.0, 4.0)'])
    plt.xlabel('Coordinates')
    plt.ylabel('Class Probabilities')
    plt.savefig(name + '/probabilities_square_experiment.png')
    plt.clf()
    #plt.show()

def square_info_eight_scores(name, classes=8):
    model = name + '/model.h5'
    loaded_model = model_load(model)
    analyser = BaseAnalyser()

    X, tensor_X = dataset_generator(-2, 4.01, -2, 4.01, 0.01)
    output = analyser.get_output(tensor_X, loaded_model, -2)

    z1_arr = [z_value[0] for z_value in output]
    z2_arr = [z_value[1] for z_value in output]
    if classes == 8:
        z3_arr = [z_value[2] for z_value in output]
        z4_arr = [z_value[3] for z_value in output]
        z5_arr = [z_value[4] for z_value in output]
        z6_arr = [z_value[5] for z_value in output]
        z7_arr = [z_value[6] for z_value in output]
        z8_arr = [z_value[7] for z_value in output]

    plt.plot(z1_arr, color='red', label='Class 1')
    plt.plot(z2_arr, color='green', label='Class 2')
    if classes == 8:
        plt.plot(z3_arr, color='black',      label='Class 3')
        plt.plot(z4_arr, color='gray',       label='Class 4')
        plt.plot(z5_arr, color='m',          label='Class 5')
        plt.plot(z6_arr, color='darksalmon', label='Class 6')
        plt.plot(z7_arr, color='tan',        label='Class 7')
        plt.plot(z8_arr, color='olivedrab',  label='Class 8')
    plt.legend()
    x_ticks = [[-2.0, 4.0], [4.0, 4.0], [4.0, -2.0], [-2.0, -2.0], [-2.0, 4.0]]
    x_tick_pos = [X.index(tick) for tick in x_ticks]
    x_tick_pos[-1] = X[1:].index([-2.0, 4.0])
    plt.xticks(x_tick_pos, [str((x[0], x[1])) for x in x_ticks])
    plt.xlabel('Coordinates')
    plt.ylabel('Class Scores')
    plt.savefig(name + '/z_values_square_experiment.png')
    plt.clf()

def white_box_drawer(name, plot_min, plot_max, max_prob, folder, layers, X, y, classes=2):
    model_name = name + '/model.h5'
    model = model_load(model_name)
    n_grid = 200
    x_plot = np.linspace(plot_min, plot_max, n_grid)
    y_plot = np.linspace(plot_min, plot_max, n_grid)

    points = []
    for xx in x_plot:
        for yy in y_plot:
            points.append((yy, xx))
    points = np.array(points)


    probs = model(points).numpy()


    if max_prob:
        z_plot = probs.max(1)
    else:
        z_plot = probs[:, 0]
    z_plot = z_plot.reshape(len(x_plot), len(y_plot)) * 100

    vmax = 100
    vmin = 50 if max_prob else 0
    if classes == 8:
        vmin = 10
        plt.contourf(x_plot, y_plot, z_plot, levels=np.linspace(10, 100, 90))
        cbar = plt.colorbar(ticks=np.linspace(vmin, vmax, 10))

        cbar.ax.set_title('confidence', fontsize=12, pad=12)
        cbar.set_ticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    else:
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
    if classes == 8:
        X2 = X[y_np.argmax(1) == 2]
        X3 = X[y_np.argmax(1) == 3]
        X4 = X[y_np.argmax(1) == 4]
        X5 = X[y_np.argmax(1) == 5]
        X6 = X[y_np.argmax(1) == 6]
        X7 = X[y_np.argmax(1) == 7]
        plt.scatter(X2[:, 0], X2[:, 1], s=20, edgecolors='black', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X3[:, 0], X3[:, 1], s=20, edgecolors='gray', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X4[:, 0], X4[:, 1], s=20, edgecolors='m', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X5[:, 0], X5[:, 1], s=20, edgecolors='darksalmon', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X6[:, 0], X6[:, 1], s=20, edgecolors='tan', facecolor='None',
                    marker='s', linewidths=0.2)
        plt.scatter(X7[:, 0], X7[:, 1], s=20, edgecolors='olivedrab', facecolor='None',
                    marker='s', linewidths=0.2)
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])
    plt.xlabel('x')
    plt.ylabel('y')

    if 'two_moons_paper' in name:
        plt.gca().add_patch(plt.Rectangle((-2.5, -2.5), 6.25, 6.25, fill=False, edgecolor='black'))
        plt.plot([0.5,3.75], [0.5,0.5], color='black')
    else:
        plt.gca().add_patch(plt.Rectangle((-2.0, -2.0), 6.0, 6.0, fill=False, edgecolor='black'))
        plt.plot([1.0, 4.0], [1.0, 1.0], color='black')

    plt.gca().set_aspect('equal', adjustable='box')
    fig = plt.gcf()
    #size = fig.get_size_inches() * fig.dpi
    fig.set_size_inches(6, 4)
    plt.savefig('{}/{}_{}_{}.pdf'.format(
        folder, layers, plot_min, plot_max), transparent=True)
    plt.clf()
    #plt.show()

def box_plotter(model, file, sets, classes=2):
    moons = MoonsDataset(classes)
    if sets == 1:
        X, y, x_test, y_test = moons.two_moons()
    else:
        X, y, x_test, y_test = moons.four_set_two_moons()
    white_box_drawer(model, -5.0, 6.0, True, model, file, X, y, classes)

if __name__ == "__main__":

    eight_moon_names = [
        'two_moons_four_paper/100d_100d_relu_800_iter_mact_int_zeros_new_fit_8_classes',
        'two_moons_four_paper/100d_100d_relu_800_iter_softmax_int_zeros_new_fit_8_classes']
    two_moons_names = [
        'two_moons_paper/100d_100d_relu_400_iter_mact_int_zeros_new_fit',
        'two_moons_paper/100d_100d_relu_400_iter_softmax_int_zeros_new_fit'
    ]
    ablation_study_names = [
        'two_moons_four_paper/100d_100d_relu_800_iter_mact_int_zeros_new_fit',
        'two_moons_four_paper/100d_100d_relu_800_iter_softmax_int_zeros_new_fit',
        'two_moons_four_paper/100d_100d_relu_800_iter_mact_int_c=1_b=0.01_not_trainable_new_fit'
    ]

    for name in eight_moon_names:
        square_info_eight_probs(name)
        square_info_eight_scores(name)
        softmax_problems_eigth_probs(name)
        softmax_problems_eigth_scores(name)
        box_plotter(name, 'boxed_plot', sets=4, classes=8)
        print("Done {}".format(name))
    for name in two_moons_names:
        square_info_two_probs(name)
        square_info_two_scores(name)
        softmax_problems_small_probs(name)
        softmax_problems_small_scores(name)
        box_plotter(name, 'boxed_plot', sets=1, classes=2)
        print("Done {}".format(name))
    for name in ablation_study_names:
        square_info_eight_scores(name, classes=2)
        square_info_eight_probs(name, classes=2)
        softmax_problems_large_probs(name)
        softmax_problems_large_scores(name)
        box_plotter(name, 'boxed_plot', sets=4, classes=2)
        print("Done {}".format(name))