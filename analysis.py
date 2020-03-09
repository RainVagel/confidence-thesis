import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import MoonsDataset, CifarDataset, MnistDataset, Cifar10GrayScale, SVHNDataset, EMnistDataset, \
    FMnistDataset, DataGenerator


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


def saved_model_tests(model_name, dataset):
    loaded_model = load_model(model_name)
    datasets = {}
    if dataset.upper() == 'MNIST':
        for data in [MnistDataset(aug=False), FMnistDataset(aug=False), EMnistDataset(aug=False),
                     Cifar10GrayScale(aug=False)]:
            train_gen = DataGenerator(data, data.n_train, False, "train", aug=['normalize'])
            test_gen = DataGenerator(data, data.n_test, False, "test", aug=['normalize'])
            x_train, y_train = train_gen.get_analysis()
            x_test, y_test = test_gen.get_analysis()
            if data.__class__.__name__ == 'MnistDataset':
                tru_x_train, tru_y_train = x_train, y_train
                tru_x_test, tru_y_test = x_test, y_test
            datasets[data.__class__.__name__] = (x_train, y_train, x_test, y_test)
    elif dataset.upper() == 'CIFAR10':
        for data in [SVHNDataset(aug=False), CifarDataset(cifar_version=10, aug=False), CifarDataset(cifar_version=100, aug=False)]: # Missing LSUN_classroom and imagenet_minus_cifar10
            x_train, y_train, x_test, y_test = data.load_dataset()
            datasets[data.__class__.__name__] = (x_train, tf.keras.utils.to_categorical(y_train, data.n_classes),
                                                 x_test, tf.keras.utils.to_categorical(y_test, data.n_classes))
    elif dataset.upper() == 'CIFAR100':
        for data in [CifarDataset(cifar_version=10, aug=False), CifarDataset(cifar_version=100, aug=False), SVHNDataset(aug=False)]: # Missing LSUN_classroom and imagenet_minus_cifar10
            x_train, y_train, x_test, y_test = data.load_dataset()
            datasets[data.__class__.__name__] = (x_train, tf.keras.utils.to_categorical(y_train, data.n_classes),
                                                 x_test, tf.keras.utils.to_categorical(y_test, data.n_classes))
    elif dataset.upper() == 'SVHN':
        for data in [SVHNDataset(aug=False), CifarDataset(cifar_version=10, aug=False), CifarDataset(cifar_version=100, aug=False)]: # Missing LSUN_classroom and imagenet_minus_cifar10
            x_train, y_train, x_test, y_test = data.load_dataset()
            datasets[data.__class__.__name__] = (x_train, tf.keras.utils.to_categorical(y_train, data.n_classes),
                                                 x_test, tf.keras.utils.to_categorical(y_test, data.n_classes))
    else:
        raise Exception("Rubbish datasets not defined for this dataset")


    analyser = BaseAnalyser()

    for key in datasets:
        x_train = datasets[key][0]
        y_train = datasets[key][1]
        x_test = datasets[key][2]
        y_test = datasets[key][3]
        tru_train_pred = loaded_model.predict(tru_x_train)
        tru_test_pred = loaded_model.predict(tru_x_test)
        tru_lbl = analyser.tru(tru_y_train)
        tru_rbs_lbl = analyser.tru(y_train)*False
        pred_train = loaded_model.predict(x_train)
        pred_test = loaded_model.predict(x_test)
        conf_tru_train = analyser.max_conf(tru_train_pred)
        conf_train = analyser.max_conf(pred_train)
        conf_test = analyser.max_conf(pred_test)
        print("Model: {}".format(model_name))
        print("MMC train, dataset: {}, value: {}".format(key, np.mean(conf_train)))
        print("MMC test, dataset: {}, value: {}".format(key, np.mean(conf_test)))
        (fpr, tpr, thresholds), auc_score = analyser.roc(tru_rbs_lbl, conf_train, tru_lbl, conf_tru_train)
        print("ROC AUC, dataset: {}, score: {}".format(key, auc_score))
        fpr95, clean_tpr95 = analyser.fpr_at_95_tpr(conf_train, conf_test)
        print("FPR at {}%, dataset: {}, score: {}".format(95, key, fpr95))


if __name__ == '__main__':
    #saved_model_tests("activation_line/paper_MNIST_lenet_softmax", "MNIST")
    #runner = LeNetRunner(True)

    #model = runner.load_model(input_shape=(28, 28, 1), num_classes=10)

    #print(model.summary())

    data = MnistDataset()
    x_test, y_test = DataGenerator(data, 128, False, mode='test', aug=['normalize']).get_analysis()

    loaded_model = load_model("tf_upgrade_2/paper_MNIST_lenet_mact.h5", custom_objects={'MActAbs': MActAbs})

    print(loaded_model.predict(x_test))
    analyser = BaseAnalyser()
    print(analyser.get_output_trial(x_test, loaded_model, 'dense_1'))
