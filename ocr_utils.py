# -*- coding: utf-8 -*-

# import easyocr # type: ignore
from pymatreader import read_mat
import pandas as pd
import numpy as np
import pytesseract


def load_matlab_format_data(mode='train'):
    if mode == 'train':
        LABELS_FILE = "./train/digitStruct.mat"
    if mode == 'test':
        LABELS_FILE = "./test/digitStruct.mat"

    data_mat = read_mat(LABELS_FILE)
    df = pd.DataFrame(data_mat)
    
    image_files = df.loc['name', 'digitStruct'] #data.iloc[1] 
    image_files  = np.array(image_files).reshape(-1,1)
    image_files = pd.DataFrame(image_files)
    bbox_and_labels = df.loc['bbox', 'digitStruct'] #data.iloc[0]
    # bbox_and_labels = np.array(bbox_and_labels).reshape(-1,1)
    # bbox_and_labels = pd.DataFrame(bbox_and_labels[0][0])
    bbox_and_labels = pd.json_normalize(bbox_and_labels)

    df = image_files.copy()
    df = df.join(bbox_and_labels)

    df = df.rename(columns={0: 'file_name'}) 
    df = df[['file_name', 'height', 'left', 'top', 'width', 'label']]
    
    return df
    # return image_files,bbox_and_labels

def get_labels(mode='train'):
    df = load_matlab_format_data(mode=mode)
    labels = df['label'].apply(lambda x: [int(i) for i in x] if isinstance(x, list) else int(x))
    return labels

def save_labels(df, mode='train'):
    if mode == 'train':
        LABELS_FILE = "./train/labels.csv"
    if mode == 'test':
        LABELS_FILE = "./test/labels.csv"
    
    df.to_csv(LABELS_FILE)


# reader = easyocr.Reader(['en'])
# def get_text(img, reader=reader):
#     # detail=0 will make readtext return a list of elements found in the image
    return reader.readtext(img, detail=0)

def read_text_tesseract(img):
    return pytesseract.image_to_string(img)