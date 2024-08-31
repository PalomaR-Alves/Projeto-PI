import cv2
import numpy as np
from skimage.filters import threshold_niblack  # type: ignore
from skimage import measure


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversão para tons de cinza
    img = cv2.equalizeHist(img)  # equaliza histograma da imagem
    return img


def blur_image(img):
    mean_ker = np.array([[1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9]])
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)


def binarize_niblack(img):
    niblack_thresh = threshold_niblack(img, window_size=25, k=0.8)
    image_binarized_niblack = (img > niblack_thresh).astype('uint8') * 255
    return image_binarized_niblack


def binarize_otsu(img):
    (T, threshInv) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return threshInv


def bilateral_filter(img):
    return cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)


def adaptive_limiar(img, mode):
    if mode == "mean":
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    if mode == "gaus":
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


def segmentation(img):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    if len(img.shape) == 3 and img.shape[2] == 3:
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Input image must be a color image")

    V = cv2.split(image_hsv)[2]
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


def preprocess_image_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    inverted = cv2.bitwise_not(morph)
    return inverted


def preprocess_image_with_connected_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output_image = np.zeros_like(img)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)

        if 100 < area < 1000 and 0.2 < aspect_ratio < 1.0:
            component_mask = (labels == i).astype("uint8") * 255
            component = component_mask[y:y + h, x:x + w]
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            output_image[y:y + h, x:x + w] = cv2.bitwise_or(output_image[y:y + h, x:x + w], component)

    inverted_output = cv2.bitwise_not(output_image)
    return inverted_output
