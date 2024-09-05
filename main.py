#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
import config
import image_utils
from ocr_utils import read_text_tesseract
# from ocr_utils import get_labels

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima"
__email__ = "kelves.nunes@ufrpe.br, palomaraissa10@gmail.com, limarbson7@gmail.com"
__programname__ = "Projeto PdI"


def __get_args():
    parser = ArgumentParser(prog=__programname__, description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-in", "--input", dest="input_dir", default=config.DATA, help="Input dir path")
    parser.add_argument("-out", "--output", dest="output_dir", default=config.OUTPUTS, help="Output dir")

    return parser.parse_args()


def process_image(img):
    print("Iniciando o processamento de imagem...")

    # Passo 1: Segmentação tradicional
    segmented_img = image_utils.segmentation(img)
    print("Segmentação tradicional realizada.")

    # Passo 2: Detecção de Bordas
    edges_img = image_utils.detect_edges(segmented_img)
    print("Detecção de bordas realizada.")

    # Passo 3: Segmentação por Regiões (Watershed)
    region_segmented_img = image_utils.region_based_segmentation(img)
    print("Segmentação por regiões realizada.")

    return segmented_img, edges_img, region_segmented_img


def main():
    args = __get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.use_ocr:
        labels = np.array([])
        # true_labels = get_labels

    for filename in os.listdir(input_dir):
        print(f"Processando imagem: {filename}")
        image_path = os.path.join(input_dir, filename)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Falha ao carregar a imagem {filename}")
            continue

        segmented_img, edges_img, region_segmented_img = process_image(img)

        # Salvando a imagem segmentada
        output_segmented_path = os.path.join(output_dir, f"segmented_{filename}")
        cv2.imwrite(output_segmented_path, segmented_img)
        print(f"Imagem segmentada salva em: {output_segmented_path}")

        # Salvando a imagem com detecção de bordas
        output_edges_path = os.path.join(output_dir, f"edges_{filename}")
        cv2.imwrite(output_edges_path, edges_img)
        print(f"Imagem com bordas detectadas salva em: {output_edges_path}")

        # Salvando a imagem segmentada por regiões
        output_region_segmented_path = os.path.join(output_dir, f"region_segmented_{filename}")
        cv2.imwrite(output_region_segmented_path, region_segmented_img)
        print(f"Imagem segmentada por regiões salva em: {output_region_segmented_path}")


if __name__ == '__main__':
    main()
