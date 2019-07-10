from pdf2image import convert_from_path
from scipy import misc
from PIL import Image
import numpy as np
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import os
from copy import copy
from image_mask import process_image


do_pdf2img = False
do_color2bw = False
do_process_image = True

# convert pdf to images
if do_pdf2img:
    for file_name in os.listdir("Data"):
        if os.path.isfile(os.path.join("Data", file_name)):
            file_path = os.path.join("Data/", file_name)

            pages = convert_from_path(file_path, 500)
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


def matprint(mat):
    fd = open("exp.txt", "w")
    for x in mat:
        for i, y in enumerate(x):
            y = 1 if y > 0 else 0
            fd.write(str(y))
        fd.write("\n")
    fd.close()


def BFS(matrix):
    matrix_copy = copy(matrix)
    Xs = [0, len(matrix[0]) - 1]
    Ys = [0, len(matrix) - 1]
    for x in Xs:
        for y in Ys:
            queue = list()
            queue.append((y, x))
            index = 0
            while index < len(queue):
                y, x = queue[index]
                if matrix[y][x] == 0:
                    matrix[y][x] = 255
                    matrix_copy[y][x] = -1
                    if y > 0 and matrix_copy[y - 1][x] != -1:
                        queue.append((y - 1, x))
                    if y > 0 and x > 0 and matrix_copy[y - 1][x - 1] != -1:
                        queue.append((y - 1, x - 1))
                    if x > 0 and matrix_copy[y][x - 1] != -1:
                        queue.append((y, x - 1))
                    if y < Ys[1] and x > 0 and matrix_copy[y + 1][x - 1] != -1:
                        queue.append((y + 1, x - 1))
                    if y < Ys[1] and matrix_copy[y + 1][x] != -1:
                        queue.append((y + 1, x))
                    if y < Ys[1] and x < Xs[1] and matrix_copy[y + 1][x + 1] != -1:
                        queue.append((y + 1, x + 1))
                    if x < Xs[1] and matrix_copy[y][x + 1] != -1:
                        queue.append((y, x + 1))
                    if y > 0 and x < Xs[1] and matrix_copy[y - 1][x + 1] != -1:
                        queue.append((y - 1, x + 1))
                index += 1
    return matrix


# resize a picture
# with open('test-image.jpeg', 'r+b') as f:
#     with Image.open(f) as image:
#         cover = resizeimage.resize_cover(image, [200, 100])
#         cover.save('test-image-cover.jpeg', image.format)

# convert black and white images to matrices
if do_process_image:
    for dir in os.listdir("Data"):
        if os.path.isdir(os.path.join("Data", dir)):
            for file in os.listdir(os.path.join("Data", dir)):
                file_path = os.path.join(os.path.join("Data", dir), file)
                process_image(file_path)
                # matrix = misc.imread(file_path)
                # sum_vect = np.sum(matrix, axis=1)# / len(matrix[0])
                # a = np.hstack((matrix.normal(size=1000),matrix.normal(loc=5, scale=2, size=1000)))
                # plt.hist(np.array(matrix).flatten(), bins='auto')  # arguments are passed to np.histogram
                # plt.show()
                # matrix = BFS(matrix)
                # misc.imsave(file_path, matrix)
                # matprint(matrix)
