import os
import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def preprocessing(img):
    image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    image_gray_equalized = cv2.equalizeHist(image_grayscale) # equaliza histograma da imagem
    return image_gray_equalized


def blur_image(img):
    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)

def segmentation(img):
    
    # image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # V = cv2.split(img)[2]
    image_gray_thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)
    labels = measure.label(image_gray_thresholded, connectivity=2, background=0)
    mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')

    # Ciclamos sobre cada etiqueta.
    for i, label in enumerate(np.unique(labels)):
        if label == 0:
            continue

        label_mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
    
        # Se um componente for grande o suficiente então o adicionamos ao resultado final
        if 300 < num_pixels < 1500:
            mask = cv2.add(mask, label_mask)

    return mask
