import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    img = cv2.equalizeHist(img) # equaliza histograma da imagem
    return img

def normalize(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    return cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

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
    (_, threshInv) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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

