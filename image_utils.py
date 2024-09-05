import cv2
import numpy as np
from skimage.filters import threshold_niblack  # type: ignore
from skimage import measure


def preprocess_image_for_ocr(img):
    print("[INFO] Iniciando o pré-processamento da imagem para OCR...")

    # Convertendo para tons de cinza
    print("[INFO] Convertendo a imagem para escala de cinza...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Removendo ruído usando filtro bilateral
    print("[INFO] Removendo ruído da imagem...")
    img_filtered = cv2.bilateralFilter(img_gray, 11, 17, 17)

    # Aplicando detecção de bordas (Canny) para realçar contornos
    print("[INFO] Aplicando detecção de bordas (Canny)...")
    edged = cv2.Canny(img_filtered, 30, 200)

    return edged


def preprocessing(img):
    print("[INFO] Convertendo a imagem para escala de cinza...")
    if len(img.shape) == 3:  # Verifica se a imagem tem 3 canais (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversão para tons de cinza
    print("[INFO] Equalizando o histograma da imagem...")
    img = cv2.equalizeHist(img)  # equaliza histograma da imagem
    return img


def blur_image(img):
    print("[INFO] Aplicando filtro de borramento (Mean Kernel)...")
    mean_ker = np.array([[1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9]])
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)


def binarize_niblack(img):
    print("[INFO] Aplicando binarização Niblack...")
    niblack_thresh = threshold_niblack(img, window_size=25, k=0.8)
    image_binarized_niblack = (img > niblack_thresh).astype('uint8') * 255
    return image_binarized_niblack


def bilateral_filter(img):
    print("[INFO] Aplicando filtro bilateral...")
    return cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)


def adaptive_limiar(img, mode):
    if mode == "mean":
        print("[INFO] Aplicando limiar adaptativo (média)...")
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    if mode == "gaus":
        print("[INFO] Aplicando limiar adaptativo (Gaussiano)...")
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


def segmentation(img, method="connected_components"):
    print(f"[INFO] Iniciando a segmentação usando o método {method}...")

    if method == "connected_components":
        print("[INFO] Segmentação por componentes conexos...")
        if len(img.shape) == 3 and img.shape[2] == 3:
            image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            V = cv2.split(image_hsv)[2]
        else:
            V = img

        image_gray_thresholded = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)
        labels = measure.label(image_gray_thresholded, connectivity=2, background=0)
        mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')

        for _, label in enumerate(np.unique(labels)):
            if label == 0:
                continue

            label_mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')
            label_mask[labels == label] = 255
            num_pixels = cv2.countNonZero(label_mask)

            if 50 < num_pixels < 500:
                mask = cv2.add(mask, label_mask)

        return mask

    elif method == "watershed":
        print("[INFO] Segmentação por Watershed...")

        # Garantir que a imagem seja BGR
        if len(img.shape) == 2 or img.shape[2] != 3:
            print("[INFO] Convertendo a imagem para BGR...")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(opening, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)

        # Transformar markers para CV_32SC1
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = np.int32(markers)

        # Aplicar watershed
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]

        return img

    elif method == "kmeans":
        print("[INFO] Segmentação por K-Means...")

        # Garantir que a imagem esteja em BGR
        if len(img.shape) == 2 or img.shape[2] != 3:
            print("[INFO] Convertendo a imagem para BGR...")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape((img.shape))
        return segmented_image

    else:
        raise ValueError(f"[ERRO] Método de segmentação desconhecido: {method}")
