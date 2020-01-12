from pathlib import Path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from dataset import MoonsDataset
from models import ModelRunner, MAct, MActModelRunner


def create_model():
    model = Sequential([
        Dense(100, input_shape=(2,)),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(2),
        MAct()
    ])

    return model


def main():
    file_name = "codebase_trials/four_moons"
    run_name = "dense100_dense100_relu_MAct"

    Path(file_name).mkdir(parents=True, exist_ok=True)
    print("Creating model!")
    model = create_model()
    print("Model created!")
    runner = MActModelRunner(file_name, run_name, 200, 2)
    optimizer = Adam(learning_rate=0.01)
    X, y, X_test, y_test = MoonsDataset().four_set_two_moons()
    runner.model_experiment(model=model, optimizer=optimizer, acet=False, X=X, y=y, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()
