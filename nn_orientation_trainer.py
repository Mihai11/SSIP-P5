import os

import cv2
import numpy as np

from nn_orientation import ImageOrientationSequence, setup_model


def train(training_folder, validation_folder):
    train_generator = ImageOrientationSequence(training_folder, batches_per_iteration=100)
    validation_generator = ImageOrientationSequence(validation_folder, batches_per_iteration=10)
    mX, my = train_generator[0]
    model = setup_model(mX, my)
    model.fit_generator(train_generator, epochs=10, validation_data=validation_generator,
                        use_multiprocessing=False, max_queue_size=10, workers=1)
    return model


def get_orientation(model, image):
    if len(image.shape) == 2:
        image = image.reshape((image.shape + (1,)))
    else:
        if image.shape[-1] != 1:
            raise Exception('Expected to have grayscale image')
    if np.max(image) > 1.:
        raise Exception('Image is expected to be normalized to 1')
    batch = image.reshape((1,) + image.shape)
    result = model.predict(batch)
    print(f'orientation_result shape {result.shape}')
    result = result[0]
    return np.argmax(result[:])


def generator_test(input_folder):
    ios = ImageOrientationSequence(input_folder)
    for i in range(10):
        X, y = ios[i]
        print(f'X {X.shape} y {y.shape}')


if __name__ == '__main__':
    # generator_test('orientation_data/train')
    model = train('orientation_data/train', 'orientation_data/validation')
    model_folder = 'orientation_model'
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'orientation.h5'))

    img = cv2.imread('orientation_data/validation/g_005.png', cv2.IMREAD_GRAYSCALE)
    img = img / 255
    result = get_orientation(model, img)
    print(result)
