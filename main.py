import tqdm as tqdm
from pdf2image import convert_from_path
from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from image_mask import *

do_pdf2img = False
do_color2bw = False
do_process_image = True

if __name__ == '__main__':
    # convert pdf to images
    if do_pdf2img:
        for file_name in os.listdir("Data"):
            if os.path.isfile(os.path.join("Data", file_name)):
                file_path = os.path.join("Data/", file_name)

                pages = convert_from_path(file_path, 300)
                for index, page in enumerate(pages):
                    picture_path = '{}/{}_out_{}.png'.format(file_path.split(".")[0], file_name.split(".")[0], index)
                    page.save(picture_path, 'PNG')
                    print('{}_out_{}.png'.format(file_name.split(".")[0], index))

    # convert colored images to black and white images
    if do_color2bw:
        for dir in os.listdir("Data"):
            if os.path.isdir(os.path.join("Data", dir)):
                for file in os.listdir(os.path.join("Data", dir)):
                    file_path = os.path.join(os.path.join("Data", dir), file)
                    image_file = Image.open(file_path)  # open colour image
                    image_file = image_file.convert('L').point(
                        lambda band: 255 if band > 240 else 0)  # convert image to black and white
                    image_file.save(file_path)

    # convert black and white images to matrices
    if do_process_image:
        for dir in os.listdir("Data"):
            if os.path.isdir(os.path.join("Data", dir)):
                for file in tqdm.tqdm(os.listdir(os.path.join("Data", dir)), desc=dir):
                    original_file_path = os.path.join(os.path.join("Data", dir), file)
                    file_path = os.path.dirname(original_file_path) + "\\processed\\" + original_file_path.split("\\")[-1]
                    if os.path.isfile(original_file_path):
                        rotated_image = get_rotated_image(original_file_path)
                        misc.imsave(file_path, rotated_image)
                        x, y, w, h = bbox_image(file_path)
                        crop_image(x, y, w, h, file_path, file_path)
                        # matrix = misc.imread(file_path)
                        # sum_vect = np.sum(matrix, axis=1)# / len(matrix[0])
                        # a = np.hstack((matrix.normal(size=1000),matrix.normal(loc=5, scale=2, size=1000)))
                        # plt.hist(np.array(matrix).flatten(), bins='auto')  # arguments are passed to np.histogram
                        # plt.show()
                        # matrix = BFS(matrix)
                        # misc.imsave(file_path, matrix)
                        # matprint(matrix)
