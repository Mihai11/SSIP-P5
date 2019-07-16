import numpy as np


def rotate_image_right_angles(img, orientation):
    """

    :param img:
    :param orientation: 0 = 0 degrees
                        1 = 90 degrees
                        2 = 180 degrees
                        3= 270 degrees
    :return:  rotated image
    """
    return np.rot90(img, k=orientation)
