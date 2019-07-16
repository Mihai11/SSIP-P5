from nn_orientation import ImageOrientationSequene


def generator_test(input_folder):
    ios = ImageOrientationSequene(input_folder)
    for i in range(10):
        X, y = ios[i]
        print(f'X {X.shape} y {y.shape}')


if __name__ == '__main__':
    generator_test('orientation_data/train')
