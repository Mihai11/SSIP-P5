import glob
import random

import cv2
import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, Conv2D, GlobalAveragePooling2D, \
    Concatenate, Dropout, TimeDistributed, Dense, GlobalAveragePooling1D, Flatten, GlobalMaxPooling2D
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, TerminateOnNaN, \
    ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Nadam

from image_utils import rotate_image_right_angles

FLOAT_TYPE = np.float32

DEFAULT_WINDOW_SIZES = [256,
                        #   512, 768,
                        # 1024,
                        ]


class ImageOrientationSequence(Sequence):

    def __init__(self, folder_name, image_extension='.png',
                 window_sizes=None, batches_per_iteration=1000,
                 batch_size=256) -> None:
        super().__init__()
        self.folder_name = folder_name
        self.window_sizes = window_sizes if window_sizes else DEFAULT_WINDOW_SIZES
        self.samples_per_iteration = batches_per_iteration
        self.batch_size = batch_size
        self.image_list = []
        self.labels_encoded = []
        for image_fn in glob.glob(f'{folder_name}/*{image_extension}', recursive=False):
            img = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE) / 255
            img = img.reshape((img.shape + (1,)))
            for rotate in range(4):
                self.image_list.append(rotate_image_right_angles(img, rotate))
                label = np.zeros((1, 4), dtype=FLOAT_TYPE)
                label[0, rotate] = 1.
                self.labels_encoded.append(label)
        print(f'Got {len(self.image_list)} rotated images')

    def __getitem__(self, index):
        window_size = random.choice(self.window_sizes)
        X = np.zeros((self.batch_size, window_size, window_size, 1), dtype=FLOAT_TYPE)
        y = np.zeros((self.batch_size, 4), dtype=FLOAT_TYPE)
        for i in range(self.batch_size):
            index = random.randint(0, len(self.image_list) - 1)
            image = self.image_list[index]
            x_image = random.randint(0, image.shape[0] - window_size - 1)
            y_image = random.randint(0, image.shape[1] - window_size - 1)
            X[i, :, :, :] = image[x_image:x_image + window_size, y_image:y_image + window_size, :]
            y[i, :] = self.labels_encoded[index]
        return X, y

    def __len__(self):
        return self.samples_per_iteration

    def on_epoch_end(self):
        super().on_epoch_end()


def setup_model(X, y):
    print(f'Setting up model with input shape {X.shape} and output shape {y.shape}')
    input = Input(shape=(None, None, X.shape[-1]), name='input_image')

    layer = input

    layer = Conv2D(32, kernel_size=3, activation='relu')(layer)
    layer = Conv2D(32, kernel_size=3, activation='relu')(layer)
    # layer = Conv2D(32, kernel_size=3, activation='relu')(layer)
    # layer = Conv2D(16, kernel_size=3, activation='relu')(layer)

    # layer = Conv2D(32, kernel_size=5, activation='relu')(layer)
    # layer = Conv2D(32, kernel_size=7, activation='relu')(layer)

    # layer = GlobalAveragePooling2D()(layer)
    layer = GlobalMaxPooling2D()(layer)

    # layer = Dense(y.shape[-1] * 2, activation='relu')(layer)

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
    train_size = 20000  # 10000
    train_window = None  # 24
    X_train_adjusted = X_train[:train_size, train_window:-train_window if train_window is not None else None,
                       train_window:-train_window if train_window is not None else None]
    y_train_adjusted = y_train[:train_size, :]
    print(f'Training on {X_train_adjusted.shape} {y_train_adjusted.shape}')
    model.fit(X_train_adjusted, y_train_adjusted,
              validation_data=(X_test, y_test), epochs=3)
