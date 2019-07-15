from image_mask import bbox_image
from image_mask import *
from scipy import misc


output_file = "processed\\Ion_Heliade_Radulescu_out_7.png"
file_path = "output+25.jpg"
file_path = "output25.jpg"
file_path = "Ion_Heliade_Radulescu_out_7.png"
rotated_image = get_rotated_image(file_path)
misc.imsave(output_file, rotated_image)
x, y, w, h = bbox_image(output_file)
crop_image(x, y, w, h, output_file, output_file)
