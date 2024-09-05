#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from skimage.filters import threshold_niblack # type: ignore
# from skimage import measure

from argparse import ArgumentParser, RawTextHelpFormatter
import image_utils

__authors__ = "Kelves Costa, Paloma Raissa, Rubson Lima "
__email__ = "kelves.nunes@ufrpe.br, palomaraissa10@gmail.com, limarbson7@gmail.com"

__programname__ = "Projeto PdI"
# rodar no terminal
# pip install opencv-python matplotlib scikit-image pillow
# para executar: python main.py -in input -out output 

def __get_args():
    parser = ArgumentParser(prog=__programname__, description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--debug", dest="debug_mode", action="store_true", help="Active debug mode")
    #parser.add_argument("-i", "--image", dest="image", required=True, help="Input image path")
    parser.add_argument("-in", "--input", dest="input_dir", required=True, help="Input dir path")
    parser.add_argument("-out", "--output", dest="output_dir", required=True, help="Output dir")
    parser.add_argument("--use-ocr", dest="use_ocr", action="store_true", help="[TODO] Use OCR.")
    parser.add_argument("-m", "--metric", dest="metrics", action="store_true", help="[TODO] Measure models performance.")

    parsed_args = parser.parse_args()

    return parsed_args

def main():
    args = __get_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    for filename in os.listdir(input_dir):
        # caminho completo do arquivo de entrada
        image_path = os.path.join(input_dir, filename)
        
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Failed to load image {filename}")
            continue

        # img = image_utils.preprocessing(img)
        # img = image_utils.bilateral_filter(img)
        # img = image_utils.blur_image(img)
        # img = image_utils.mean_blur(img)
        # img = image_utils.median_blur(img)
        # img = image_utils.binarize_niblack(img)
        # img = image_utils.binarize_otsu(img)
        # img = image_utils.adaptive_limiar(img, 'mean')
        # img = image_utils.preprocess_image_with_connected_components(img)
        # img = image_utils.connected_components_segmentation(img)
        # img = image_utils.find_contours(img)
        # img = image_utils.canny_edges(img)
        # img = image_utils.sobel_edges(img)
        # img = image_utils.watershed_segmentation(img)
        # img = image_utils.kmeans_segmentation(img)
        # img = image_utils.preprocess_image_for_ocr(img)
        # img = image_utils.find_contours(img)
        img = image_utils.kmeans_segmentation(img)
        

        # caminho completo para salvar a imagem processada
        output_path = os.path.join(output_dir, filename)
        
        # salva a imagem processada
        cv2.imwrite(output_path, img)
    

    # reader = easyocr.Reader(['en'])  # Inicializa o EasyOCR

"""
    image_original = cv2.imread(args.image) # carrega a imagem em BGR

    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    image_gray_equalized = cv2.equalizeHist(image_grayscale) # equaliza histograma da imagem

    # filtro bilateral para redução de ruído
    # d: diâmetro da vizinhança
    # sigmaColor: variância para filtro de intensidade, aumentar suaviza pixels com diferenças de cor maiores
    # sigmaSpace: variância para filtro espacial, aumentar suaviza pixels mais distantes
    image_bilateral_filtered = cv2.bilateralFilter(image_gray_equalized, d=15, sigmaColor=75, sigmaSpace=75)

    # cálculo do limiar 
    # window_size: quantidade de pixels na vizinhança que terão seu desvio padrão e média calculados
    # window_size menor: captura mais detalhes porém pode ter mais ruídos
    # window_size maior: binarização mais suave mas pode perder detalhes menores
    # k: com valor menor, o limiar local será mais próx da média dos pixels locais
    #    com valor maior, pixels precisam ser mais "brilhantes" pra não serem tidos como fundo
    niblack_thresh = threshold_niblack(image_bilateral_filtered, window_size=25, k=0.8)
    image_binarized_niblack = (image_bilateral_filtered > niblack_thresh).astype('uint8') * 255 # binarização com o limiar

    # binarizacao de imagem usando a limiarização adaptativa

    # args (em ordem): imagem, valor max atribuido a pixels q passam o limiar, indicação de forma de calculo do
    # limiar, operação para manter binarização, tamanho da vizinhança, valor subtraido da constante C
    # sobre o valor subtraido: se positivo, subtrai mais do limiar e facilita a classificação de um pixel como branco
    # se negativo, subtrai menos ou adiciona, dificultando que um pixel seja classificado como branco
    adap_media_thresh = cv2.adaptiveThreshold(image_bilateral_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    adap_gauss_thresh = cv2.adaptiveThreshold(image_bilateral_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

    
    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    V = cv2.split(image_hsv)[2]
    image_hsv_thresholded = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)
    labels = measure.label(image_hsv_thresholded, connectivity=2, background=0)
    mask = np.zeros(image_hsv_thresholded.shape, dtype='uint8')

    cv2.imshow('Imagen binaria antes da tags', image_hsv_thresholded)

    # Ciclamos sobre cada etiqueta.
    for i, label in enumerate(np.unique(labels)):
        if label == 0:
            continue

        label_mask = np.zeros(image_hsv_thresholded.shape, dtype='uint8')
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
    
        # Se um componente for grande o suficiente então o adicionamos ao resultado final
        if 300 < num_pixels < 1500:
            mask = cv2.add(mask, label_mask)
    
    # Exibição das regiões de interesse detectadas previamente
    cv2.imshow('Regiao de interesse', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
    
if __name__ == '__main__':
     main()
