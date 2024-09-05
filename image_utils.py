import os
import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    img = cv2.equalizeHist(img) # equaliza histograma da imagem
    return img


def blur_image(img): # útil?
    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)

def mean_blur(img): # filtro da média com máscara 5x5
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    return img

def median_blur(img):
    img = cv2.medianBlur(img, 15) # filtro da mediana com máscara 21x21

def binarize_niblack(img):
    # cálculo do limiar 
    # window_size: quantidade de pixels na vizinhança que terão seu desvio padrão e média calculados
    # window_size menor: captura mais detalhes porém pode ter mais ruídos
    # window_size maior: binarização mais suave mas pode perder detalhes menores
    # k: com valor menor, o limiar local será mais próx da média dos pixels locais
    #    com valor maior, pixels precisam ser mais "brilhantes" pra não serem tidos como fundo
    niblack_thresh = threshold_niblack(img, window_size=5, k=0.2) # 0.2 foi o melhor valor encontrado
    image_binarized_niblack = (img > niblack_thresh).astype('uint8') * 255 # binarização com o limiar
    return image_binarized_niblack

def binarize_otsu(img):
    (T, threshInv) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
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
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    if mode == "gauss":
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def connected_components_segmentation(img):
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
        if 50 < num_pixels < 6000:
            mask = cv2.add(mask, label_mask)

    return mask

def find_contours(img):
    bin = binarize_otsu(img)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Desenhar os contornos na imagem original
    image_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contours, contours, -1, (0,255,0), 2)
    return image_contours

def canny_edges(img):
    img = cv2.Canny(img, 100, 300)
    return img

def watershed_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return img

def kmeans_segmentation(img, k=8):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

# def color_segmentation(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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