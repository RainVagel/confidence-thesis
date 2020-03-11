import sys
from math import ceil
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf

from analysis import BaseAnalyser
from dataset import MoonsDataset, CifarDataset, MnistDataset, Cifar10GrayScale, SVHNDataset, EMnistDataset, \
    FMnistDataset, DataGenerator
from models import ModelRunner, MAct, MActModelRunner, MActAbs, CustomHistory, CifarModelRunner, LeNetRunner,\
    ResNetSmallRunner, lmd_added_loss


def create_relu_model():
    model = Sequential([
        Dense(100, input_shape=(2,)),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(2),
        MActAbs(b_trainable=False)
    ])

    return model


def create_tanh_model():
    model = Sequential([
        Dense(100, input_shape=(2,)),
        Activation('tanh'),
        Dense(100),
        Activation('tanh'),
        Dense(100),
        Activation('tanh'),
        Dense(100),
        Activation('tanh'),
        Dense(2),
        MActAbs(b_trainable=False)
    ])

    return model


def folder_creater(file_name):
    Path(file_name).mkdir(parents=True, exist_ok=True)


def main():
    file_name = "codebase_trials/four_moons/trial"
    run_name = "dense100_dense100_relu_MAct_saved"

    Path(file_name).mkdir(parents=True, exist_ok=True)
    print("Creating model!")
    model = create_model()
    print("Model created!")
    runner = MActModelRunner(model, file_name, run_name, iterations=1000, dim=2)
    optimizer = Adam(learning_rate=0.01)
    moons_dataset = MoonsDataset()
    X, y, X_test, y_test = moons_dataset.four_set_two_moons(n_samples=1000)
    #print(X)
    runner.model_experiment(optimizer=optimizer, acet=False, X=X, y=y, X_test=X_test, y_test=y_test, batch_size=128, inter_plots=True)
    runner.save_model()

    #loaded_model = load_model("codebase_trials/four_moons/dense100_dense100_relu_MAct_saved")

    #print(loaded_model.summary())


def run_for_trial(model_function, n_iterations, file_name, run_name):
    print(model_function)
    if model_function == 'create_relu_model':
        model = create_relu_model()
    elif model_function == 'create_tanh_model':
        model = create_tanh_model()
    else:
        raise Exception("Such model does not exist!")
    folder_creater(file_name)
    runner = MActModelRunner(model, file_name, run_name, iterations=n_iterations, dim=2)
    optimizer = Adam(learning_rate=0.001)
    moons_dataset = MoonsDataset()
    X, y, X_test, y_test = moons_dataset.four_set_two_moons(n_samples=1000)
    runner.model_experiment(optimizer=optimizer, acet=False, X=X, y=y, X_test=X_test, y_test=y_test, batch_size=128,
                            inter_plots=True)
    runner.save_model()


def cifar_example():
    """
    An example function that shows how to use the CifarModelRunner and run and analyse a cifar model from start to finish.
    :return:
    """
    file_name = "cifar_trial/example"
    run_name = "keras_cifar_10"

    Path(file_name).mkdir(parents=True, exist_ok=True)
    analyser = BaseAnalyser()

    # Create the model runner class. Model and dim arguments are not important
    cifar_runner = CifarModelRunner("something", file_name, run_name, iterations=3, dim=0)

    # Create the Cifar dataset class
    cifar_data = CifarDataset()
    # Load the cifar-10 dataset
    x_train, y_train, x_test, y_test = cifar_data.load_dataset()

    # Load the default Keras cifar-10 model
    cifar_runner.load_keras_cifar10_model(x_train, mact=True)
    # Load the default optimizer
    optimizer = cifar_runner.get_default_optimizer()
    # Compile the model
    cifar_runner.compile_model(optimizer)
    # Run the model experiment
    cifar_runner.model_experiment(optimizer, x_train, y_train, x_test, y_test, batch_size=128)
    # Save the model
    cifar_runner.save_model(file_name, run_name)
    # Plot the history from callbacks
    analyser.plot_history(file_name, run_name, cifar_runner.get_history())


def paper_example():
    le = LeNetRunner(mact=True)
    model = le.load_model(input_shape=(28, 28, 1), num_classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = tf.keras.datasets.mnist.load_data()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    X_train = np.reshape(X_train, (60000, 28, 28, 1))
    X_test = np.reshape(X_test, (10000, 28, 28, 1))

    # Convert training and test labels to one hot matrices
    Y_train = tf.keras.utils.to_categorical(Y_train_orig, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test_orig, 10)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model.fit(X_train, Y_train, epochs=25, batch_size=32)

    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))


def mnist_train():
    x_train, y_train, x_test, y_test = MnistDataset().load_dataset()
    lr = 0.001
    batch_size = 128
    n_epochs = 100

    # Learning rate scheduler based on the code from the article
    n_iter_per_epoch = x_train.shape[0] // batch_size
    decay1 = round(0.5 * n_iter_per_epoch * n_epochs)
    decay2 = round(0.75 * n_iter_per_epoch * n_epochs)
    decay3 = round(0.90 * n_iter_per_epoch * n_epochs)
    lr_decay_n_updates = [decay1, decay2, decay3]
    lr_decay_coefs = [lr, lr / 10, lr / 100, lr / 1000]

    step = tf.Variable(0, trainable=False)
    #boundaries = [50, 75, 90]
    #values = [0.001, 0.0001, 0.00001, 0.000001]
    learning_rate_fn = PiecewiseConstantDecay(
        lr_decay_n_updates, lr_decay_coefs)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    le = LeNetRunner(mact=True)
    model = le.load_model(input_shape=(28, 28, 1), num_classes=10)
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=batch_size)

    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))


def scheduler(epoch, lr):
    if epoch == 50:
        return lr / 10
    elif epoch == 75:
        return lr / 10
    elif epoch == 90:
        return lr / 10
    else:
        return lr


def paper_train(dataset, model_name, folder_name, name=None, mact=True, n_epochs=100):
    batch_size = 128

    print(mact)
    print("Creating file")
    folder_creater(folder_name)
    print("File created")

    print("Loading model")
    if model_name == 'resnet':
        runner = ResNetSmallRunner(mact=mact)
    elif model_name == 'lenet':
        runner = LeNetRunner(mact=mact)
    else:
        raise Exception('Unsupported model')
    print("Model loaded")

    print("Loading dataset")
    if dataset == 'SVHN':
        train_aug = ['randomcrop', 'normalize']
        test_aug = ['normalize']
        data = SVHNDataset()
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=10)
    elif dataset == 'MNIST':
        train_aug = ['randomcrop', 'normalize']
        test_aug = ['normalize']
        data = MnistDataset()
        model = runner.load_model(input_shape=(28, 28, 1), num_classes=10)
    elif dataset == 'CIFAR10':
        train_aug = ['horizontalflip', 'randomcrop', 'normalize']
        test_aug = ['normalize']
        data = CifarDataset()
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=10)
    elif dataset == 'CIFAR100':
        train_aug = ['horizontalflip', 'randomcrop', 'normalize']
        test_aug = ['normalize']
        data = CifarDataset(cifar_version=100)
        model = runner.load_model(input_shape=(32, 32, 3), num_classes=100)
    else:
        raise Exception('Unsupported dataset for training')

    steps_epoch = int(data.n_train/batch_size)
    val_steps = int(data.n_test/batch_size)

    train_gen = DataGenerator(data, batch_size, True, aug=train_aug)
    test_gen = DataGenerator(data, batch_size, False, aug=test_aug)
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

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        LearningRateScheduler(scheduler, verbose=1),
        CustomHistory()
    ]

    print("STarting training")
    H = model.fit_generator(train_gen, steps_per_epoch=steps_epoch, validation_steps=val_steps,
                            validation_data=test_gen, epochs=n_epochs, callbacks=callbacks, workers=24,
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
    preds = model.evaluate_generator(test_gen)
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

    # Saving the b and c learnable parameters
    with open(params_file_name, 'w') as filehandle:
        filehandle.write("B;C\n")
        for b, c in zip(H.history["b"], H.history["c"]):
            filehandle.write("{};{}\n".format(b, c))


def mnist_yield_trial():
    folder_name = 'mnist_yield_trial'
    print("Creating file")
    folder_creater(folder_name)
    print("File created")

    print("Loading model")
    runner = LeNetRunner(mact=True)

    print("Loading dataset")

    data = MnistDataset()

    train_aug = ['randomcrop', 'normalize']
    test_aug = ['normalize']

    train_gen = DataGenerator(data, 128, True, aug=train_aug)
    test_gen = DataGenerator(data, 128, False, aug=test_aug)

    #train_gen = MnistDataset().load_dataset()
    model = runner.load_model(input_shape=(28, 28, 1), num_classes=10)
    print("Dataset loaded")

    lr = 0.001
    n_epochs = 2

    optimizer = Adam(lr)


    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    callbacks = [
        LearningRateScheduler(scheduler, verbose=1)
    ]

    print("STarting training")
    H = model.fit_generator(train_gen, callbacks=callbacks, epochs=n_epochs, validation_data=test_gen)
    print("Model trained")

    print("Saving model")
    dataset = 'mnist'
    model_name = 'generator_trial'
    runner.save_model(model, folder_name, 'paper_{}_{}'.format(dataset, model_name))
    print("Model saved")

    print("Evaluating model")
    preds = model.evaluate_generator(test_gen)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    N = 2
    print(H.history.keys())
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def rbs_generator(true_data, rbs_data_lst):
    datasets = {}
    data = true_data
    tru_x_test, tru_y_test = DataGenerator(data, data.n_test, False, "test", aug=['normalize']).get_analysis()
    for data in rbs_data_lst:
        test_gen = DataGenerator(data, data.n_test, False, "test", aug=['normalize'])
        x_test, y_test = test_gen.get_analysis()
        datasets[data.__class__.__name__] = (x_test, y_test)
    return tru_x_test, tru_y_test, datasets


def saved_model_tests(model_name, dataset):
    loaded_model = load_model(model_name, custom_objects={'MActAbs': MActAbs})

    if dataset.upper() == 'MNIST':
        trained_dataset = MnistDataset(aug=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [FMnistDataset(aug=False), EMnistDataset(aug=False),
                                                          Cifar10GrayScale(aug=False)])
    elif dataset.upper() == 'CIFAR10':
        trained_dataset = CifarDataset(cifar_version=10, aug=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [SVHNDataset(aug=False),
                                                          CifarDataset(cifar_version=100, aug=False)]) # Missing LSUN_classroom and imagenet_minus_cifar10
    elif dataset.upper() == 'CIFAR100':
        trained_dataset = CifarDataset(cifar_version=100, aug=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [SVHNDataset(aug=False),
                                                          CifarDataset(cifar_version=10, aug=False)]) # Missing LSUN_classroom and imagenet_minus_cifar10
    elif dataset.upper() == 'SVHN':
        trained_dataset = SVHNDataset(aug=False)
        tru_x_test, tru_y_test, datasets = rbs_generator(trained_dataset,
                                                         [CifarDataset(cifar_version=100, aug=False),
                                                          CifarDataset(cifar_version=10, aug=False)])  # Missing LSUN_classroom and imagenet_minus_cifar10
    else:
        raise Exception("Rubbish datasets not defined for this dataset")

    analyser = BaseAnalyser()

    tru_test_pred = loaded_model.predict(tru_x_test)
    tru_lbl = analyser.tru(tru_y_test)
    conf_tru_test = analyser.max_conf(tru_test_pred)
    print("Model: {}".format(model_name))
    print("MMC, dataset: {}, value: {}".format(trained_dataset.__class__.__name__, np.mean(conf_tru_test)))

    for key in datasets:
        rbs_x_test = datasets[key][0]
        rbs_y_test = datasets[key][1]
        tru_rbs_lbl = analyser.tru(rbs_y_test)*False
        rbs_pred_test = loaded_model.predict(rbs_x_test)
        conf_rbs_test = analyser.max_conf(rbs_pred_test)
        print("MMC, dataset: {}, value: {}".format(key, np.mean(conf_rbs_test)))
        (fpr, tpr, thresholds), auc_score = analyser.roc(tru_rbs_lbl, conf_rbs_test, tru_lbl, conf_tru_test)
        print("ROC AUC, dataset: {}, score: {}".format(key, auc_score))
        fpr95, clean_tpr95 = analyser.fpr_at_95_tpr(conf_tru_test, conf_rbs_test)
        print("FPR at {}%, dataset: {}, score: {}".format(95, key, fpr95))


def layer_output_analyser(model, dataset):
    analyser = BaseAnalyser()
    x_test, _ = DataGenerator(dataset, dataset.n_test, False, "test", aug=['normalize']).get_analysis()
    loaded_model = load_model(model, custom_objects={'MActAbs': MActAbs})
    output_layer = analyser.get_output(x_test, loaded_model, -1)
    np.savetxt(model.split(".")[0] + "_output_layer.csv", output_layer, delimiter=";")
    output_layer = analyser.get_output(x_test, loaded_model, -2)
    np.savetxt(model.split(".")[0] + "_pre_output_layer.csv", output_layer, delimiter=";")

if __name__ == "__main__":
    #trained_dataset = MnistDataset(aug=False)
    #layer_output_analyser("regularization/paper_MNIST_lenet_mact.h5", trained_dataset)
    #saved_model_tests("regularization/paper_MNIST_lenet_softmax.h5", "MNIST")
    #loaded_model = load_model("regularization/paper_MNIST_lenet_softmax.h5", custom_objects={'MActAbs': MActAbs})
    #print(loaded_model.summary())

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
    paper_train(dataset_inp, model_inp, folder_name, name_inp, mact_inp, n_epochs)

    #loaded_model = load_model("mact_trials/paper_CIFAR10_resnet_softmax.h5", custom_objects={'MActAbs': MActAbs})
    #print(loaded_model.summary())

     #data = MnistDataset()
     #x_test, y_test = DataGenerator(data, 128, False, mode='test', aug=['normalize']).get_analysis()

     #loaded_model = load_model("mact_trials/paper_MNIST_lenet_mact.h5", custom_objects={'MActAbs': MActAbs})

     #print(loaded_model.predict(x_test))
     #analyser = BaseAnalyser()
     #print(analyser.get_output(x_test, loaded_model, -2))

    #model = create_tanh_model()
    #moons_dataset = MoonsDataset()
    #X, y, X_test, y_test = moons_dataset.four_set_two_moons(n_samples=1000)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #history = CustomHistory()
    #model.fit(X, y, validation_data=(X_test, y_test), batch_size=128, epochs=10, verbose=1, callbacks=[history])
    #print(history.history)

    # Create the Cifar dataset class
    #cifar_data = CifarDataset()
    # Load the cifar-10 dataset
    #x_train, y_train, x_test, y_test = cifar_data.load_dataset()
    #cifar_labels = cifar_data.load_label_names()
    #print(cifar_labels)

    #loaded_model = load_model("cifar_trial/example/keras_cifar_10")

    #predictions = loaded_model.predict(x_test)

    #x_test = x_test[:10]


    #analyser = BaseAnalyser()
    #print(analyser.mean_max_conf(loaded_model, x_test))

    #print(analyser._tru(y_test))

    #loaded_model = load_model("four_moons/tanh_modded_5/d100_d100_d100_d100_d2_tanh_MActAbs")

    #analyser = BaseAnalyser()

    #moons_dataset = MoonsDataset()

    #X, y, X_test, y_test = moons_dataset.four_set_two_moons(n_samples=1000)

    #analyser.single_output_plot(model=loaded_model, layer=-2, file_name="trololo", layers="lubub", plot_min=0.0, plot_max=2.0)

