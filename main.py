#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
import image_utils
from ocr_utils import read_text_tesseract
# from ocr_utils import get_labels

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima "
__email__ = "kelves.nunes@ufrpe.br, palomaraissa10@gmail.com, limarbson7@gmail.com"

__programname__ = "Projeto PdI"
# rodar no terminal
# pip install opencv-python matplotlib scikit-image pillow
def __get_args():
    parser = ArgumentParser(prog=__programname__, description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--debug", dest="debug_mode", action="store_true", help="Active debug mode")
    parser.add_argument("-in", "--input", dest="input_dir", required=True, help="Input dir path")
    parser.add_argument("-out", "--output", dest="output_dir", required=False, help="Output dir")
    parser.add_argument("-s", "--save-output",dest="save_output", action="store_true", help="[TODO] Save output images.")
    parser.add_argument("--use-ocr", dest="use_ocr", action="store_true", help="[TODO] Use OCR.")
    parser.add_argument("-m", "--metric", dest="metrics", action="store_true", help="[TODO] Measure models performance.")

    parsed_args = parser.parse_args()

    return parsed_args

def main():
    args = __get_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.use_ocr:
        labels = np.array([])
        # true_labels = get_labels

    for filename in os.listdir(input_dir):
        # caminho completo do arquivo de entrada
        image_path = os.path.join(input_dir, filename)
        
        img = cv2.imread(image_path)
        
        img = image_utils.preprocessing(img)
        img = image_utils.normalize(img)
        img = image_utils.bilateral_filter(img)
        img = image_utils.adaptive_limiar(img, mode='gaus')
        # img = image_utils.segmentation(img)

        if args.use_ocr:
            text = read_text_tesseract(img)
            print(text)
            # np.append(labels, label_read)

        
        if args.save_output: # and not args.output_dir
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
    
if __name__ == '__main__':
     main()
