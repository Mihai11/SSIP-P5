import argparse
import glob
import os
from multiprocessing.pool import Pool

import cv2
import numpy as np
import wand
import tqdm
from wand.color import Color
from wand.image import Image as WandImage

from config import Config

pool = None


def extract_pdf_images(p):
    """
    extracts images from a pdf file and returns the list of image file names (and pdf name)
    :param pdf_file:
    :param args:
    :return:
    """
    pdf_file, args = p
    name = os.path.splitext(os.path.split(pdf_file)[1])[0]
    print(f'Processing {name}')

    crt_work_folder = os.path.join(args.work_folder, '001_images', name)
    os.makedirs(crt_work_folder, exist_ok=True)

    image_list = glob.glob(f'{crt_work_folder}/*{Config.IMAGE_EXTENSION}', recursive=False)
    if not image_list:
        # pdf not processed
        with open(pdf_file, 'rb') as fpdf:
            with WandImage(file=fpdf, resolution=Config.RESOLUTION, depth=8) as img:
                print('pdf_pages = ', len(img.sequence), pdf_file)
                for page_num, crt_img in tqdm.tqdm(list(enumerate(img.sequence)), desc=f'Processing {name}'):
                    fn_start = str(page_num).zfill(3)
                    fn = os.path.join(crt_work_folder, fn_start + Config.IMAGE_EXTENSION)
                    if not os.path.exists(fn):
                        with WandImage(
                                resolution=(Config.RESOLUTION, Config.RESOLUTION), depth=8) as dst_image:
                            # converted.background_color = Color('white')
                            # converted.alpha_channel = 'remove'
                            # converted.save(filename=fn)
                            with WandImage(crt_img) as im2:
                                im2.background_color = Color('white')
                                im2.alpha_channel = 'remove'
                                dst_image.sequence.append(im2)
                                # dst_image.resolution=(resolution,resolution)
                                dst_image.units = 'pixelsperinch'
                                dst_image.background_color = Color('white')
                                dst_image.save(filename=fn)
    image_list = glob.glob(f'{crt_work_folder}/*{Config.IMAGE_EXTENSION}', recursive=False)
    return [('/'.join((name, os.path.splitext(os.path.split(image_fn)[1])[0])), image_fn) for image_fn in image_list]


def rotate_image(p):
    name, image_fn, args = p

    rotated_fn = os.path.join(args.work_folder, '002_rotated', name) + Config.IMAGE_EXTENSION
    os.makedirs(os.path.basename(rotated_fn), exist_ok=True)
    if not os.path.exists(rotated_fn):
        original = cv2.imread(image_fn, cv2.IMREAD_COLOR)
        # TODO: generate rotated image
        rotated = original
        cv2.imwrite(rotated_fn, rotated)

    return (name, rotated_fn)


def process_pdf_folder(args):
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.work_folder, exist_ok=True)
    pdf_list = glob.glob(f'{args.input_folder}/**/*.pdf', recursive=True)
    print(f'Found {len(pdf_list)} pdf files')
    global pool
    if pool is None:
        pool = Pool()
    image_list = []
    for r in pool.imap_unordered(extract_pdf_images, ((pdf_file, args) for pdf_file in pdf_list)):
        image_list.extend(r)
    print(f'Found {len(image_list)} images')
    print(f'First item is  {image_list[0]}')
    rotated_images = [r for r in tqdm.tqdm(pool.imap(rotate_image, ((p[0], p[1], args) for p in image_list)),
                                           total=len(image_list), desc='rotating images')]
    print(f'First item in rotated images is {rotated_images[0]}')

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Extracts scanned pages from pdf files"
    )
    parser.add_argument("--input_folder",
                        help="folder containing input pdf files", type=str,
                        default='Data')

    parser.add_argument("--output_folder",
                        help='folder containing output files',
                        default='Ouput')

    parser.add_argument("--work_folder",
                        help='intermediate files', default='Work')

    parser.set_defaults()

    parser.print_help()

    args = parser.parse_args()
    print(args)

    process_pdf_folder(args)
