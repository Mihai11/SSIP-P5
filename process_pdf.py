import argparse
import glob
import os
from multiprocessing.pool import Pool

pool = None


def process_pdf_folder(args):
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.work_folder, exist_ok=True)
    pdf_list = glob.glob(f'{args.input_folder}/**/*.pdf', recursive=True)
    print(f'Found {len(pdf_list)} pdf files')
    global pool
    if pool is None:
        pool = Pool()

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
