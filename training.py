import sys
from math import ceil
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import os
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf

from analysis import BaseAnalyser
from dataset import DataGenerator
from models import RadialSoftmax, CustomHistory,  LeNetRunner,\
    ResNetSmallRunner
import dataset_old as data_old


def folder_creater(file_name):
    Path(file_name).mkdir(parents=True, exist_ok=True)


def scheduler(epoch, lr):
    if epoch == 50:
        return lr / 10
    elif epoch == 75:
        return lr / 10
    elif epoch == 90:
        return lr / 10
    else:
        return lr


def paper_train_torch(dataset, model_name, folder_name, name=None, mact=True, n_epochs=100, activ=None):
    batch_size = 128

    print(mact)
    print("Creating file")
    folder_creater(folder_name)
    print("File created")

    print("Loading model")
    if model_name == 'resnet':
        runner = ResNetSmallRunner(mact=mact)
    elif model_name == 'lenet':
        runner = LeNetRunner(mact=mact, activation=activ)
    else:
        raise Exception('Unsupported model')
    print("Model loaded")

    print("Loading dataset")
    if dataset == 'SVHN':
        dataset_class = data_old.SVHN(128, True)
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=10)
    elif dataset == 'MNIST':
        dataset_class = data_old.MNIST(128, True)
        model = runner.load_model(input_shape=(28, 28, 1), num_classes=10)
    elif dataset == 'CIFAR10':
        dataset_class = data_old.CIFAR10(128, True)
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=10)
    elif dataset == 'CIFAR100':
        dataset_class = data_old.CIFAR100(128, True)
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=100)
    else:
        raise Exception('Unsupported dataset for training')

    steps_epoch = int(dataset_class.n_train/batch_size)
    val_steps = int(dataset_class.n_test/batch_size)

    train_gen = dataset_class.get_train_batches(n_batches='all', shuffle=True)
    test_gen = dataset_class.get_test_batches(n_batches='all', shuffle=False)
    print("Dataset loaded")

    if dataset == 'MNIST':
        lr = 0.001
    elif dataset in ('SVHN', 'CIFAR10', 'CIFAR100'):
        lr = 0.1
    else:
        raise Exception("Unsupported dataset for training!")

    if dataset == 'MNIST':
        optimizer = Adam(lr)
    elif dataset in ('SVHN', 'CIFAR10', 'CIFAR100'):
        optimizer = SGD(lr, momentum=0.9)
    else:
        raise Exception("Unsupported dataset for training!")

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if mact:
        callbacks = [
            LearningRateScheduler(scheduler, verbose=1),
            CustomHistory()
        ]
    else:
        callbacks = [
            LearningRateScheduler(scheduler, verbose=1)
        ]

    print("STarting training")
    H = model.fit_generator(train_gen, steps_per_epoch=steps_epoch, epochs=n_epochs, callbacks=callbacks, workers=1,
                            max_queue_size=30)
    print("Model trained")

    print("Saving model")
    if name is None:
        runner.save_model(model, folder_name, 'paper_{}_{}'.format(dataset, model_name))
        plot_name = '{}/paper_{}_{}_acc_plot.png'.format(folder_name, dataset, model_name)
        params_file_name = '{}/paper_{}_{}_params.csv'.format(folder_name, dataset, model_name)
    else:
        runner.save_model(model, folder_name, 'paper_{}_{}_{}'.format(dataset, model_name, name))
        plot_name = '{}/paper_{}_{}_{}_acc_plot.png'.format(folder_name, dataset, model_name, name)
        params_file_name = '{}/paper_{}_{}_{}_params.csv'.format(folder_name, dataset, model_name, name)
    print("Model saved")

    print("Evaluating model")
    preds = model.evaluate_generator(test_gen, steps=val_steps)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_name)

    if mact:
        # Saving the b and c learnable parameters
        with open(params_file_name, 'w') as filehandle:
            filehandle.write("B;C\n")
            for b, c in zip(H.history["b"], H.history["c"]):
                filehandle.write("{};{}\n".format(b, c))


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


def layer_output_analyser(model, dataset_name, dataset):
    analyser = BaseAnalyser()
    x_test, _ = DataGenerator(dataset, dataset.n_test, False, "test", aug=['normalize']).get_analysis()
    loaded_model = load_model(model, custom_objects={'RadialSoftmax': RadialSoftmax})
    output_layer = analyser.get_output(x_test, loaded_model, -1)
    np.savetxt("{}_{}_output_layer.csv".format(model.split(".")[0], dataset_name), output_layer, delimiter=";")
    output_layer = analyser.get_output(x_test, loaded_model, -2)
    np.savetxt("{}_{}_pre_output_layer.csv".format(model.split(".")[0], dataset_name), output_layer, delimiter=";")

if __name__ == "__main__":
    all_model_analysis('paper_extra')

    dataset_inp = sys.argv[1]
    model_inp = sys.argv[2]
    folder_name = sys.argv[3]
    mact_inp = sys.argv[4]
    mact_inp = True if mact_inp.lower() == 'true' else False
    try:
        name_inp = sys.argv[5]
    except Exception:
        name_inp = None
    n_epochs = int(sys.argv[6])
    activ = sys.argv[7]

    paper_train_torch(dataset_inp, model_inp, folder_name, name_inp, mact_inp, n_epochs, activ)

