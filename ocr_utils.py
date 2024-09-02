# -*- coding: utf-8 -*-

import easyocr # type: ignore
from pymatreader import read_mat
import pandas as pd
import numpy as np

# ====== freaking annoying data format
def load_matlab_format_data(mode='train'):
    if mode == 'train':
        LABELS_FILE = "./train/digitStruct.mat"
    if mode == 'test':
        LABELS_FILE = "./test/digitStruct.mat"

    data_mat = read_mat(LABELS_FILE)
    df = pd.DataFrame(data_mat)
    df = df.iloc[1]
    df = np.array(df).reshape(-1,1)
    df = pd.DataFrame(df[0][0])
    return df

def get_labels(mode='train'):
    df = load_matlab_format_data(mode)
    labels = df['label'].apply(lambda x: [int(i) for i in x] if isinstance(x, list) else int(x))
    return labels

class Ocr():
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def get_text(self, img):
        # detail=0 will make readtext return a list of elements found in the image
        return self.reader.readtext(img, detail=0)

    def compare_results(self):
        pass