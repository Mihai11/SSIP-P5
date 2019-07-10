from image_mask import process_image
from PIL import Image

file = "-374-rau_out_4.png"
# image_file = Image.open(file)  # open colour image
# image_file = image_file.convert('L')
# image_file.save("p_" + file)
# process_image("p_" + file)
process_image(file)