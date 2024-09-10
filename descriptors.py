import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Cálculo para os momentos Hu
def compute_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


# Função para calcular LBP e criar o histograma
def compute_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Função para extrair características usando Momentos de Hu e LBP
def extract_features(images):
    features = []
    for img in images:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Converte de RGB para BGR
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Converte para cinza
        hu_moments = compute_hu_moments(img) # Calcula os momentos de hu
        lbp_hist = compute_lbp(img) # Calcula o histograma do lbp
        features.append(np.hstack([hu_moments, lbp_hist])) # Cada imagem possui uma lista com os vetores de hu e lbp adicionada na lista de features
    return np.array(features)
