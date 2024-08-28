import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def blur_image(img):
    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)

def segmentation(img):
    
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = cv2.split(image_hsv)[2]
    image_hsv_thresholded = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)
    labels = measure.label(image_hsv_thresholded, connectivity=2, background=0)
    mask = np.zeros(image_hsv_thresholded.shape, dtype='uint8')

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

    return mask
