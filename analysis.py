import csv
import pickle
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import roc_auc_score, roc_curve

from models import RadialSoftmax
import dataset_old as data_old


class BaseAnalyser:
    def __init__(self, m_act=True):
        self.m_act = m_act

    def get_output(self, X, model, layer):
        layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
        output = layer_output([X])[0]
        return output

    def get_output_trial(self, X, model, layer_name):
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_model.predict(X)
        return intermediate_output

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

    def plot_history(self, file_name, layers, history):
        for key in history.history:
            plt.plot(history.history[key])
            plt.ylabel(key)
            plt.xlabel("Epoch")
            plt.title(key)
            plt.savefig('{}/{}_{}.png'.format(file_name, layers, key))
            plt.clf()

    def conf_labeller(self, preds, true, labels, file_name, layers):
        header = "prediction;prediction_confidence;true_label;true_label_confidence"
        for label in labels:
            header += ";{}_confidence".format(label, label)
        with open(file_name + "/" + layers + ".csv", 'w', newline='') as myfile:
            myfile.write(header)
            myfile.write("\n")
            for pred_index in range(len(preds)):
                prediction = preds[pred_index]
                pred_label_idx = np.argmax(preds[pred_index])
                true_idx = np.where(true[pred_index] == 1.)[0][0]
                myfile.write(labels[pred_label_idx])
                myfile.write(";" + str(prediction[pred_label_idx]))
                myfile.write(";" + labels[true_idx])
                myfile.write(";" + str(prediction[true_idx]))
                for pred_cycle_idx in range(len(prediction)):
                    myfile.write(";" + str(prediction[pred_cycle_idx]))
                myfile.write("\n")

    def _max_conf(self, preds):
        return np.max(preds, axis=1)

    # def tru(self, a):
    #     return np.isin(a[:, 0], [0, 1])

    def tru(self, a):
        return np.isin(a[:, 0], [0, 1])

    def roc(self, true_set, conf_set, true_clean, conf_clean):
        tru_with_clean = np.concatenate([true_set, true_clean])
        conf_with_clean = np.concatenate([conf_set, conf_clean])
        return roc_curve(tru_with_clean, conf_with_clean, pos_label=True), roc_auc_score(tru_with_clean, conf_with_clean)

    def max_conf(self, preds):
        #count = len(x_test)
        #sum = 0
        #for pred in preds:
        #    sum += float(pred[np.argmax(pred)])
        #return round(sum / count, 2)
        return np.max(preds, axis=1)

    def fpr_at_95_tpr(self, conf_t, conf_f):
        TPR = 95
        PERC = np.percentile(conf_t, 100-TPR)
        FP = np.sum(conf_f >= PERC)
        FPR = np.sum(conf_f >= PERC) / len(conf_f)
        return FPR, PERC


def plot_auroc(fpr, tpr, label, title):
    plt.plot(fpr, tpr, label=label)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    #plt.show()


def title_maker(name):
    splitted_name = name.split("_")
    if 'mact' in splitted_name[1]:
        other_part = ' Radial Softmax'
    else:
        other_part = ' Softmax'
    return splitted_name[0] + other_part + ' Area Under Curve'


def compute_analysis(file):
    with open(file, 'rb') as handle:
        df = pickle.load(handle)
    final_results = dict()
    for k, v in df.items():
        inner_result = dict()
        for k2, v2 in v.items():
            most_inner_result = dict()
            most_inner_result['mmc'] = np.mean(v2['mmc'])
            most_inner_result['fpr95'] = np.mean(v2['fpr95'])
            most_inner_result['fpr'] = [np.mean(el) for el in zip(*v2['fpr'])]
            most_inner_result['tpr'] = [np.mean(el) for el in zip(*v2['tpr'])]
            most_inner_result['auroc'] = np.mean(v2['auroc'])
            inner_result[k2] = most_inner_result
        final_results[k] = inner_result

    for trained_set, rubbish_sets in final_results.items():
        print('Trained Set')
        print(trained_set)
        for rbs_set, rbs_set_values in rubbish_sets.items():
            print('Rubbish Set')
            print(rbs_set)
            print('mmc: ', rbs_set_values['mmc'])
            print('auroc: ', rbs_set_values['auroc'])
            print('fpr95: ', rbs_set_values['fpr95'])
            label = rbs_set + ', area: ' + str(round(rbs_set_values['auroc'], 3))
            plot_auroc(rbs_set_values['fpr'], rbs_set_values['tpr'], label, title_maker(trained_set))
        plt.savefig('exps_paper/' + trained_set + '_auc.png')
        plt.clf()

    #plot_auroc(most_inner_result['fpr'], most_inner_result['tpr'])

def rbs_generator(true_data, rbs_data_lst):
    datasets = {}
    tru_x_test, tru_y_test = true_data.get_analysis()
    #tru_x_test, tru_y_test = DataGenerator(data, data.n_test, False, "test", aug=['normalize']).get_analysis()
    for data in rbs_data_lst:
        #test_gen = DataGenerator(data, data.n_test, False, "test", aug=['normalize'])
        #x_test, y_test = test_gen.get_analysis()
        x_test, y_test = data.get_analysis()
        y_test = tf.one_hot(y_test, data.n_classes)
        datasets[data.__class__.__name__] = (x_test, y_test)
    return tru_x_test, tf.one_hot(tru_y_test, true_data.n_classes), datasets


def saved_model_tests(model_name, dataset):
    loaded_model = load_model(model_name, custom_objects={'RadialSoftmax': RadialSoftmax})

    if dataset.upper() == 'MNIST':
        trained_dataset = data_old.MNIST(batch_size=10000, augm_flag=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [data_old.FMNIST(batch_size=10000, augm_flag=False),
                                                          data_old.EMNIST(batch_size=10000, augm_flag=False),
                                                          data_old.CIFAR10Grayscale(batch_size=10000, augm_flag=False)])
    elif dataset.upper() == 'CIFAR10':
        trained_dataset = data_old.CIFAR10(batch_size=10000, augm_flag=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [data_old.SVHN(batch_size=26032, augm_flag=False),
                                                          data_old.CIFAR100(batch_size=10000, augm_flag=False),
                                                          data_old.LSUNClassroom(batch_size=300, augm_flag=False)
                                                          ])
    elif dataset.upper() == 'CIFAR100':
        trained_dataset = data_old.CIFAR100(batch_size=10000, augm_flag=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [data_old.SVHN(batch_size=26032, augm_flag=False),
                                                          data_old.CIFAR10(batch_size=10000, augm_flag=False),
                                                          data_old.LSUNClassroom(batch_size=300, augm_flag=False)
                                                          ])
    elif dataset.upper() == 'SVHN':
        trained_dataset = data_old.SVHN(batch_size=26032, augm_flag=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [data_old.CIFAR100(batch_size=10000, augm_flag=False),
                                                          data_old.CIFAR10(batch_size=10000, augm_flag=False),
                                                          data_old.LSUNClassroom(batch_size=300, augm_flag=False)
                                                          ])
    else:
        raise Exception("Rubbish datasets not defined for this dataset")

    analyser = BaseAnalyser()

    tru_test_pred = loaded_model.predict(tru_x_test)
    tru_lbl = analyser.tru(tru_y_test)
    conf_tru_test = analyser.max_conf(tru_test_pred)
    print("Model: {}".format(model_name))
    print("MMC, dataset: {}, value: {}".format(dataset, np.mean(conf_tru_test)))

    calculated_values = dict()

    for key in datasets:
        rbs_x_test = datasets[key][0]
        rbs_y_test = datasets[key][1]
        tru_rbs_lbl = analyser.tru(rbs_y_test)*False
        rbs_pred_test = loaded_model.predict(rbs_x_test)
        conf_rbs_test = analyser.max_conf(rbs_pred_test)
        mmc = np.mean(conf_rbs_test)
        print("MMC, dataset: {}, value: {}".format(key, mmc))
        (fpr, tpr, thresholds), auc_score = analyser.roc(tru_rbs_lbl, conf_rbs_test, tru_lbl, conf_tru_test)
        print("ROC AUC, dataset: {}, score: {}".format(key, auc_score))
        fpr95, clean_tpr95 = analyser.fpr_at_95_tpr(conf_tru_test, conf_rbs_test)
        print("FPR at {}%, dataset: {}, score: {}".format(95, key, fpr95))
        calculated_values[key] = {'mmc': mmc, 'fpr': fpr, 'tpr': tpr, 'fpr95': fpr95, 'auroc': auc_score}
    return calculated_values


def name_getter(name):
    splitted = name.split('.')
    splitted_2 = splitted[0].split('_')
    return '{}_{}'.format(splitted_2[1], splitted_2[-1])

def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d

def all_model_analysis(folder):
    files_list = os.walk(folder)
    results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for stuff in files_list:
        if '/' in stuff[0]:
            item = next((item for item in stuff[2] if '.h5' in item), None)
            file_path = '{}/{}'.format(stuff[0], item)
            calculated_values = saved_model_tests(file_path, item.split('_')[1])
            model_name = name_getter(item)
            for k, v in calculated_values.items():
                for k2, v2 in v.items():
                    results_dict[model_name][k][k2].append(v2)

    results_dict = to_dict(results_dict)

    with open('extras_analysis.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle)

    #print(files_list)

if __name__ == '__main__':

    method = sys.argv[1]
    file = sys.argv[2]
    if method.lower() == 'analysis':
        all_model_analysis(file)
    elif method.lower() == 'compute':
        compute_analysis('all_analysis.pickle')
    else:
        raise Exception("Unrecognised input!")
