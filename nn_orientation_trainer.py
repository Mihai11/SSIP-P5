from nn_orientation import ImageOrientationSequence, setup_model


def train(training_folder, validation_folder):
    train_generator = ImageOrientationSequence(training_folder, batches_per_iteration=100)
    validation_generator = ImageOrientationSequence(validation_folder, batches_per_iteration=10)
    mX, my = train_generator[0]
    model = setup_model(mX,my)
    model.fit_generator(train_generator,epochs=3,validation_data=validation_generator)


def generator_test(input_folder):
    ios = ImageOrientationSequence(input_folder)
    for i in range(10):
        X, y = ios[i]
        print(f'X {X.shape} y {y.shape}')


if __name__ == '__main__':
    # generator_test('orientation_data/train')
    train('orientation_data/train', 'orientation_data/validation')
