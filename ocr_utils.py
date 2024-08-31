# -*- coding: utf-8 -*-

import easyocr # type: ignore

reader = easyocr.Reader(['en'])


def get_text(img):
    # detail=0 will make readtext return a list of elements found in the image
    return reader.readtext(img, detail=0)


