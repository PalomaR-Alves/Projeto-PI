import os
import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    img = cv2.equalizeHist(img) # equaliza histograma da imagem
    return img


def blur_image(img):
    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)

def blur(img):
        return cv2.GaussianBlur(img, (3, 3), 0)

def binarize_niblack(img):
    # cálculo do limiar 
    # window_size: quantidade de pixels na vizinhança que terão seu desvio padrão e média calculados
    # window_size menor: captura mais detalhes porém pode ter mais ruídos
    # window_size maior: binarização mais suave mas pode perder detalhes menores
    # k: com valor menor, o limiar local será mais próx da média dos pixels locais
    #    com valor maior, pixels precisam ser mais "brilhantes" pra não serem tidos como fundo
    niblack_thresh = threshold_niblack(img, window_size=25, k=0.8)
    image_binarized_niblack = (img > niblack_thresh).astype('uint8') * 255 # binarização com o limiar
    return image_binarized_niblack

def binarize_otsu(img):
    (T, threshInv) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return threshInv

def bilateral_filter(img):
    # filtro bilateral para redução de ruído
    # d: diâmetro da vizinhança
    # sigmaColor: variância para filtro de intensidade, aumentar suaviza pixels com diferenças de cor maiores
    # sigmaSpace: variância para filtro espacial, aumentar suaviza pixels mais distantes
    return cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)

def adaptive_limiar(img, mode):
    # binarizacao de imagem usando a limiarização adaptativa

    # args (em ordem): imagem, valor max atribuido a pixels q passam o limiar, indicação de forma de calculo do
    # limiar, operação para manter binarização, tamanho da vizinhança, valor subtraido da constante C
    # sobre o valor subtraido: se positivo, subtrai mais do limiar e facilita a classificação de um pixel como branco
    # se negativo, subtrai menos ou adiciona, dificultando que um pixel seja classificado como branco
    if mode == "mean":
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    
    if mode == "gaus":
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def segmentation(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = cv2.split(image_hsv)[2]
    image_gray_thresholded = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)
    labels = measure.label(image_gray_thresholded, connectivity=2, background=0)
    mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')

    # Ciclamos sobre cada etiqueta.
    for _, label in enumerate(np.unique(labels)):
        if label == 0:
            continue

        label_mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
    
        # Se um componente for grande o suficiente então o adicionamos ao resultado final
        if 50 < num_pixels < 500:
            mask = cv2.add(mask, label_mask)

    return mask

def preprocess_image_for_ocr(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply adaptive thresholding or Otsu's thresholding for binarization
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove small noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Invert the image to make digits white on black background
    inverted = cv2.bitwise_not(morph)

    return inverted

def preprocess_image_with_connected_components(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to get binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create an output image to draw on
    output_image = np.zeros_like(img)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        
        # Filter components based on size and aspect ratio
        if 100 < area < 1000 and 0.2 < aspect_ratio < 1.0:
            # Extract the component
            component_mask = (labels == i).astype("uint8") * 255
            component = component_mask[y:y+h, x:x+w]
            
            # Optionally draw bounding box (for visualization)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Combine component into the final output
            output_image[y:y+h, x:x+w] = cv2.bitwise_or(output_image[y:y+h, x:x+w], component)

    # Invert the image (if needed for OCR)
    inverted_output = cv2.bitwise_not(output_image)

    return inverted_output