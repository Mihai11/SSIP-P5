import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, Conv2D, GlobalAveragePooling2D, \
    Concatenate, Dropout, TimeDistributed, Dense, GlobalAveragePooling1D
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, TerminateOnNaN, \
    ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Nadam

FLOAT_TYPE = np.float32


def setup_model(X, y):
    print(f'Setting up model with input shape {X.shape} and output shape {y.shape}')
    input = Input(shape=(None, None, X.shape[-1]), name='input_image')

    layer = input

    layer = Conv2D(32, kernel_size=3, activation='relu')(layer)
    layer = Conv2D(32, kernel_size=5, activation='relu')(layer)
    layer = Conv2D(16, kernel_size=7, activation='relu')(layer)

    layer = GlobalAveragePooling2D()(layer)

    layer = Dense(y.shape[-1], activation='softmax')(layer)

    output = layer

    model = Model(inputs=input, outputs=output)

    model.compile(
        optimizer=Nadam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()
    return model


if __name__ == '__main__':
    from keras.datasets import mnist
    from keras.utils import to_categorical

    # download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 28, 28, 1) / 255
    X_test = X_test.reshape(10000, 28, 28, 1) / 255

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = setup_model(X_train, y_train)

    print(f'Input shape {X_train.shape}')
    # train the model
    train_size = 10000  # 10000
    train_window = 2  # None  # 24
    X_train_adjusted = X_train[:train_size, train_window:-train_window, train_window:-train_window]
    y_train_adjusted = y_train[:train_size, :]
    print(f'Training on {X_train_adjusted.shape} {y_train_adjusted.shape}')
    model.fit(X_train_adjusted, y_train_adjusted,
              validation_data=(X_test, y_test), epochs=3)
