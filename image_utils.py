import cv2
import numpy as np

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure


def segmentation(img):
    print("Segmentando a imagem...")

    # Verifica se a imagem é colorida (3 canais)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Converte para tons de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Aplica a segmentação baseada em binarização Otsu
    _, segmented_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Segmentação concluída.")
    return segmented_img
  
  
def blur(img):
        return cv2.GaussianBlur(img, (3, 3), 0)


def detect_edges(img):
    print("Detectando bordas na imagem...")

    # Verifica se a imagem é colorida (3 canais)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Converte para tons de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Aplica a detecção de bordas usando o algoritmo Canny
    edges = cv2.Canny(gray, 100, 200)

    print("Detecção de bordas concluída.")
    return edges


def region_based_segmentation(img):
    print("Segmentando a imagem por regiões usando Watershed...")

    # Converte a imagem para tons de cinza, se não for já em tons de cinza
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Desfoca a imagem para remover ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplica a binarização Otsu para obter a imagem binária
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove pequenas regiões brancas usando morfologia
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilation para obter o fundo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distância para encontrar o foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Obtenha a área desconhecida
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcação de conectividade
    _, markers = cv2.connectedComponents(sure_fg)

    # Adiciona 1 a todas as marcas para que o fundo seja 1 em vez de 0
    markers = markers + 1

    # Marca a área desconhecida como 0
    markers[unknown == 0] = 0

    # Aplica o algoritmo de Watershed
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]  # Marca as bordas com vermelho

    print("Segmentação por regiões concluída.")
    return img_color


def isolate_number(img):
    print("Isolando números...")
    # Verifica se a imagem tem 3 canais (colorida)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Converte para tons de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Aplica binarização para isolar objetos de interesse
    _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontra contornos na imagem binarizada
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Obtém o retângulo delimitador para cada contorno
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filtra áreas muito pequenas
            isolated_img = img[y:y + h, x:x + w]
            print(f"Número isolado com tamanho {w}x{h}")
            return isolated_img

    print("Nenhum número encontrado.")
    return img  # Retorna a imagem original se não encontrar nada



def preprocessing(img):
    print("Aplicando pré-processamento (conversão para tons de cinza)...")
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Normaliza a imagem
    normalized_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    print("Pré-processamento concluído.")
    return normalized_img


def blur_image(img):
    print("Aplicando desfoque Gaussian para reduzir ruído...")
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    print("Desfoque aplicado.")
    return blurred_img


def binarize_otsu(img):
    print("Aplicando binarização Otsu...")
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Binarização concluída.")
    return binary_img

