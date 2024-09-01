import cv2
import numpy as np
from skimage.filters import threshold_niblack
from skimage import measure


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img


def blur_image(img):
    mean_ker = np.array([[1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9],
                         [1 / 9, 1 / 9, 1 / 9]])
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def median_blur(img):
    return cv2.medianBlur(img, 5)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)


def binarize_niblack(img):
    niblack_thresh = threshold_niblack(img, window_size=25, k=0.8)
    return (img > niblack_thresh).astype('uint8') * 255


def high_pass_filter(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def segmentation(img):
    # Método de componentes conexos (já implementado)
    pass


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


def kmeans_segmentation(img, k=2):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def contour_based_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img
