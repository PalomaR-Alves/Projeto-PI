import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_niblack
from PIL import Image

# rodar no terminal
# pip install opencv-python
# pip install matplotlib

# carrega a imagem em BGR
image = cv2.imread('C:caminho-ate-projeto\\Projeto-PI\\1.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converte a imagem para tons de cinza

eq_img = cv2.equalizeHist(gray_image) # equaliza histograma da imagem

# filtro bilateral para redução de ruído
# d: diâmetro da vizinhança
# sigmaColor: variância para filtro de intensidade, aumentar suaviza pixels com diferenças de cor maiores
# sigmaSpace: variância para filtro espacial, aumentar suaviza pixels mais distantes
bf_img = cv2.bilateralFilter(eq_img, d=15, sigmaColor=75, sigmaSpace=75)

# cálculo do limiar 
# window_size: quantidade de pixels na vizinhança que terão seu desvio padrão e média calculados
# window_size menor: captura mais detalhes porém pode ter mais ruídos
# window_size maior: binarização mais suave mas pode perder detalhes menores
# k: com valor menor, o limiar local será mais próx da média dos pixels locais
#    com valor maior, pixels precisam ser mais "brilhantes" pra não serem tidos como fundo
niblack_thresh = threshold_niblack(bf_img, window_size=25, k=0.8)
binary_image = (bf_img > niblack_thresh).astype('uint8') * 255 # binarização com o limiar

# exibe as imagens
cv2.imshow('Imagem Original', image)
cv2.imshow('Imagem em Tons de Cinza', gray_image)
cv2.imshow('Imagem com Filtro Bilateral', bf_img)
cv2.imshow('Imagem Equalizada', eq_img)
cv2.imshow('Imagem Binarizada com Niblack', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
