#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
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
    parser.add_argument("--use-ocr", dest="use_ocr", action="store_true", help="[TODO] Use OCR.")
    parser.add_argument("-m", "--metric", dest="metrics", action="store_true",
                        help="[TODO] Measure models performance.")
    return parser.parse_args()


def process_image(img, args, method):
    print(f"[INFO] Processando imagem usando o método de segmentação: {method}...")

    if args.use_ocr:
        img = image_utils.preprocess_image_for_ocr(img)

    if args.metrics:
        # Implementar medição de desempenho se necessário
        pass

    img = image_utils.preprocessing(img)
    img = image_utils.blur_image(img)
    img = image_utils.binarize_niblack(img)

    segmented_img = image_utils.segmentation(img, method=method)

    return segmented_img


def main():
    args = __get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    segmentation_methods = ["connected_components", "watershed", "kmeans"]

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to load image {filename}")
            continue

        for method in segmentation_methods:
            processed_img = process_image(img, args, method)

            output_filename = f"{os.path.splitext(filename)[0]}_{method}.png"
            output_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(output_path, processed_img)
            print(f"[INFO] Imagem processada salva em: {output_path}")


if __name__ == '__main__':
    main()
