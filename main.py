#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import statistics
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
from torchmetrics.text import CharErrorRate

import image_utils
from data_utils import load_csv_data

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima"
__email__ = "kelves.nunes@ufrpe.br, palomaraissa10@gmail.com, limarbson7@gmail.com"
__programname__ = "Projeto PdI"

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

def validate(y_read, y_target):
    cer = CharErrorRate()
    return cer(y_read, y_target)

def main():
    args = __get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.use_ocr:
        from ocr_utils import readtext
        labels = []

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            pass
        
        h = w = 100
        img = cv2.resize(img, (h, w))

        img = image_utils.high_pass(img)
        img = image_utils.kmeans_segmentation(img)
        img = image_utils.preprocessing(img)
        img = image_utils.normalize(img)
        
        # img = image_utils.bhpf(img)


        if args.use_ocr:
            text_from_img = readtext(img, reader='easyocr')
            text = '' if text_from_img == [] else min(text_from_img, key=len)
            labels.append(text)

        if args.save_output:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)

    if args.use_ocr:
        all_data = load_csv_data()
        all_labels = all_data['label']

        err = []
        for idx in range(len(labels)):
            text_read = labels[idx]
            text_target = all_labels[idx]
            err.append(validate(text_read, text_target))
            # print(f'label: {text_read}\t target: {text_target}\t error: {validate(text_read, text_target)}')

        print(f'The avarage error is {statistics.fmean(err)}')

if __name__ == '__main__':
    main()
