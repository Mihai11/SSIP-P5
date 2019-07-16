import argparse
import glob
import os
from collections import defaultdict
from multiprocessing.pool import Pool

import cv2
import tqdm
from PIL import Image
from fpdf import FPDF
from wand.color import Color
from wand.image import Image as WandImage

from config import Config

from image_mask import get_rotated_image, bbox_image, crop_image

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


def autoorient_image(p):
    name, image_fn, args = p

    autoorient_fn = os.path.join(args.work_folder, '002_autoorient', name) + Config.IMAGE_EXTENSION
    os.makedirs(os.path.dirname(autoorient_fn), exist_ok=True)
    if not os.path.exists(autoorient_fn):
        original = cv2.imread(image_fn, cv2.IMREAD_COLOR)
        autoorient = original  # TODO: call NN
        cv2.imwrite(autoorient_fn, autoorient)

    return (name, autoorient_fn)


def deskew_image(p):
    name, image_fn, args = p

    deskew_fn = os.path.join(args.work_folder, '003_deskew', name) + Config.IMAGE_EXTENSION
    os.makedirs(os.path.dirname(deskew_fn), exist_ok=True)
    if not os.path.exists(deskew_fn):
        # original = cv2.imread(image_fn, cv2.IMREAD_COLOR)
        rotated = get_rotated_image(image_fn)
        cv2.imwrite(deskew_fn, rotated)

    return (name, deskew_fn)


def extract_page_image(p):
    name, image_fn, args = p

    page_fn = os.path.join(args.work_folder, '004_extract_page', name) + Config.IMAGE_EXTENSION
    os.makedirs(os.path.dirname(page_fn), exist_ok=True)
    if not os.path.exists(page_fn):
        x, y, w, h = bbox_image(image_fn)
        crop_image(x, y, w, h, image_fn, page_fn)

    return (name, page_fn)


def create_pdf(p):
    args, name, page_list = p
    page_list = sorted(page_list)
    print(f'Generating pdf {name}')
    pdf_fn = os.path.join(args.output_folder, name + '.pdf')
    os.makedirs(os.path.dirname(pdf_fn), exist_ok=True)

    image = Image.open(page_list[0])
    width, height = image.size

    pdf = FPDF(unit="pt", format=[width, height])
    pdf.add_page()
    for image_fn in tqdm.tqdm(page_list, desc=f'Generating {name}.pdf'):
        pdf.image(image_fn)

    pdf.output(pdf_fn, "F")


def process_pdf_folder(args, callback_GUI=None):
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.work_folder, exist_ok=True)
    if os.path.isfile(args.input_folder):
        pdf_list = [args.input_folder]
    else:
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

    if callback_GUI is not None:
        callback_GUI(1)

    oriented_images = [r for r in tqdm.tqdm(pool.imap(autoorient_image, ((p[0], p[1], args) for p in image_list)),
                                            total=len(image_list), desc='orienting images')]

    if callback_GUI is not None:
        callback_GUI(2)

    deskew_images = [r for r in tqdm.tqdm(pool.imap(deskew_image, ((p[0], p[1], args) for p in oriented_images)),
                                          total=len(image_list), desc='deskewing images')]
    if callback_GUI is not None:
        callback_GUI(3)

    page_images = [r for r in tqdm.tqdm(pool.imap(extract_page_image, ((p[0], p[1], args) for p in deskew_images)),
                                        total=len(image_list), desc='extracting pages')]
    if callback_GUI is not None:
        callback_GUI(4)

    final_images = page_images

    # group images by pdf name
    grp_name = defaultdict(list)
    for name, fn in final_images:
        grp_name[name.split('/')[0]].append(fn)

    _ = [r for r in tqdm.tqdm(pool.map(create_pdf, ((args, name, file_list)
                                                    for name, file_list in grp_name.items())),
                              total=len(grp_name), desc='Generating output pdf')]
    if callback_GUI is not None:
        callback_GUI(5)


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
