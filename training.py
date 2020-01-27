import sys
from pathlib import Path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from analysis import BaseAnalyser
from dataset import MoonsDataset
from models import ModelRunner, MAct, MActModelRunner, MActAbs


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
    #model = create_model()
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


if __name__ == "__main__":
    #main()
    model_function = sys.argv[1]
    n_iterations = int(sys.argv[2])
    file_name = sys.argv[3]
    run_name = sys.argv[4]
    run_for_trial(model_function, n_iterations, file_name, run_name)

    #loaded_model = load_model("four_moons/tanh_modded_5/d100_d100_d100_d100_d2_tanh_MActAbs")

    #analyser = BaseAnalyser()

    #moons_dataset = MoonsDataset()

    #X, y, X_test, y_test = moons_dataset.four_set_two_moons(n_samples=1000)

    #analyser.single_output_plot(model=loaded_model, layer=-2, file_name="trololo", layers="lubub", plot_min=0.0, plot_max=2.0)

