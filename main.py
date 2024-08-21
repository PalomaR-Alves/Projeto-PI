#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_niblack # type: ignore

from argparse import ArgumentParser, RawTextHelpFormatter

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima "
__email__ = "kelves.nunes@ufrpe.br, <TBD>, <TBD>"

__programname__ = "Projeto PdI"
# rodar no terminal
# pip install opencv-python matplotlib scikit-image pillow
def __get_args():
    parser = ArgumentParser(
        prog=__programname__,
        description="",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image",
        dest="image",
        required=True,
        help="Input image path",
    )
    parser.add_argument(
    "--ouput",
    dest="output",
    help="Output image path",
)

    parsed_args = parser.parse_args()

    return parsed_args

def main():
    args = __get_args()
    
    image = cv2.imread(args.image) # carrega a imagem em BGR

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converte a imagem para tons de cinza

    eq_img = cv2.equalizeHist(gray_image) # equaliza histograma da imagem

    # filtro bilateral para redução de ruído
    # d: diâmetro da vizinhança
    # sigmaColor: variância para filtro de intensidade, aumentar suaviza pixels com diferenças de cor maiores
    # sigmaSpace: variância para filtro espacial, aumentar suaviza pixels mais distantes
    bf_img = cv2.bilateralFilter(eq_img, d=15, sigmaColor=75, sigmaSpace=75)

    # cálculo do limiar 
    # window_size: quantidade de pixels na vizinhança que terão seu desvio padrão e média calculados
    # window_size menor: captura mais detalhes porém pode ter mais ruídos
    # window_size maior: binarização mais suave mas pode perder detalhes menores
    # k: com valor menor, o limiar local será mais próx da média dos pixels locais
    #    com valor maior, pixels precisam ser mais "brilhantes" pra não serem tidos como fundo
    niblack_thresh = threshold_niblack(bf_img, window_size=25, k=0.8)
    binary_image = (bf_img > niblack_thresh).astype('uint8') * 255 # binarização com o limiar

    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    image_blured = cv2.filter2D(binary_image, cv2.CV_8U, mean_ker)
    cv2.imshow('Imagem borrada', image_blured)

    # Subamostragem da imagem borrada
    # image_samp_blur = image_blured[::2,::2,:]
    # cv2.imshow('Imagem borrada (subamostragem)', image_samp_blur)

    # exibe as imagens
    # cv2.imshow('Imagem Original', image)
    # cv2.imshow('Imagem em Tons de Cinza', gray_image)
    # cv2.imshow('Imagem com Filtro Bilateral', bf_img)
    # cv2.imshow('Imagem Equalizada', eq_img)
    cv2.imshow('Imagem Binarizada com Niblack', binary_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
     main()