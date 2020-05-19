import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from models import CustomHistory,  LeNetRunner,\
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


if __name__ == "__main__":

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
    activ = None if activ == '' else activ

    paper_train_torch(dataset_inp, model_inp, folder_name, name_inp, mact_inp, n_epochs, activ)

