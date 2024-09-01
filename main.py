import os
import cv2
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
import config
import image_utils

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima"
__email__ = "kelves.nunes@ufrpe.br, palomaraissa10@gmail.com, limarbson7@gmail.com"

__programname__ = "Projeto PdI"


def __get_args():
    parser = ArgumentParser(prog=__programname__, description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--debug", dest="debug_mode", action="store_true", help="Active debug mode")
    parser.add_argument("-in", "--input", dest="input_dir", default=config.DATA, help="Input dir path")
    parser.add_argument("-out", "--output", dest="output_dir", default=config.OUTPUTS, help="Output dir")
    parser.add_argument("--use-ocr", dest="use_ocr", action="store_true", help="Use OCR for text recognition.")
    parser.add_argument("-m", "--metric", dest="metrics", action="store_true", help="Measure models performance.")
    parser.add_argument("--segmentation", dest="segmentation_method", choices=["connected", "watershed", "kmeans", "contours"], default="connected", help="Segmentation method to use.")
    parser.add_argument("--blur", dest="blur_method", choices=["mean", "gaussian", "median", "bilateral"], default="mean", help="Blurring method to use.")
    parser.add_argument("--high-pass", dest="high_pass_filter", action="store_true", help="Apply high-pass filter.")
    return parser.parse_args()


def process_image(img, args):
    if args.use_ocr:
        img = image_utils.preprocess_image_for_ocr(img)

    if args.metrics:
        # Implementar medição de desempenho se necessário
        pass

    img = image_utils.preprocessing(img)

    # Aplica o método de borramento selecionado
    if args.blur_method == "mean":
        img = image_utils.blur_image(img)
    elif args.blur_method == "gaussian":
        img = image_utils.gaussian_blur(img)
    elif args.blur_method == "median":
        img = image_utils.median_blur(img)
    elif args.blur_method == "bilateral":
        img = image_utils.bilateral_filter(img)

    # Aplica o filtro passa-alta se selecionado
    if args.high_pass_filter:
        img = image_utils.high_pass_filter(img)

    # Aplica o método de segmentação selecionado
    if args.segmentation_method == "connected":
        img = image_utils.segmentation(img)
    elif args.segmentation_method == "watershed":
        img = image_utils.watershed_segmentation(img)
    elif args.segmentation_method == "kmeans":
        img = image_utils.kmeans_segmentation(img)
    elif args.segmentation_method == "contours":
        img = image_utils.contour_based_segmentation(img)

    return img


def main():
    args = __get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to load image {filename}")
            continue

        processed_img = process_image(img, args)

        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, processed_img)


if __name__ == '__main__':
    main()
