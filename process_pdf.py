import argparse
import glob
import os
from multiprocessing.pool import Pool

import wand
from wand.color import Color
from wand.image import Image as WandImage

from config import Config

pool = None


def extract_pdf_images(pdf_file, args):
    with open(pdf_file, 'rb') as fpdf:
        with WandImage(file=fpdf, resolution=Config.RESOLUTION, depth=8) as img:
            print('pdf_pages = ', len(img.sequence), pdf_file)
            for page_num, crt_img in enumerate(img.sequence):
                fn_start = str(page_num).zfill(3)
                fn = fn_start + Config.IMAGE_EXTENSION
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


def process_pdf_folder(args):
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.work_folder, exist_ok=True)
    pdf_list = glob.glob(f'{args.input_folder}/**/*.pdf', recursive=True)
    print(f'Found {len(pdf_list)} pdf files')
    global pool
    if pool is None:
        pool = Pool()
    extract_pdf_images(pdf_list[0], args)

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
