# -*- coding: utf-8 -*-

import easyocr # type: ignore
import pytesseract

reader = easyocr.Reader(['en'])
def read_with_easyocr(img, reader=reader):
    # detail=0 will make readtext return a list of elements found in the image
    return reader.readtext(img, detail=0)

def read_with_tesseract(img):
    return pytesseract.image_to_string(img)

def readtext(img,reader):
    if reader == "easyocr":
        text = read_with_easyocr(img)
    if reader == "tesseract":
        text = read_with_tesseract(img)

    return text
