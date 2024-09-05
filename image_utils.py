import cv2
import numpy as np
import itertools

from skimage.filters import threshold_niblack # type: ignore
from skimage import measure

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversão para tons de cinza
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img) # equaliza histograma da imagem
    return img

def sobel(img):
    img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # horizontal pass
    img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # vertical pass
    return img

def laplacian(img, c=1):
    if c==1:
        laplacian_kernelA0 = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]]) # Using c = +1
        moon_laplacianA0 = cv2.filter2D(img, cv2.CV_16S, laplacian_kernelA0) # Using c = +1
        res = img + c * moon_laplacianA0
    if c==(-1):
        laplacian_kernelA1 = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]]) # Using c = -1
        moon_laplacianA1 = cv2.filter2D(img, cv2.CV_16S, laplacian_kernelA1) # Using c = -1
        res = img + c * moon_laplacianA1
    
    return res

def normalize(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    return cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

def log_img(img):
    img = np.clip(img, 1e-6, None)
    # Apply log transformation method
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))

    # Specify the data type so that
    # float value will be converted to int
    # log_image = np.array(log_image, dtype = np.uint8)
    return log_image

def gamma(x, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    y = cv2.LUT(x, lookUpTable)
    return y

def gamma_img(img):
    gamma_param = 5.0
    img = gamma(img, gamma_param)

def blur_image(img):
    # Definição do filtro (máscara, kernel) de borramento
    mean_ker = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    #Execução do filtro através da biblioteca OpenCV (cv2)
    return cv2.filter2D(img, cv2.CV_8U, mean_ker)


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

def high_pass(img):
    blur = cv2.GaussianBlur(img,(31,31),0)
    filtered = img - blur
    filtered = filtered + 127*np.ones(img.shape, np.uint8)
    cv2.imwrite('output.jpg', filtered)

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

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

#Definição da métrica de distância (euclidiana)
def D(u,v,P,Q):
    return np.sqrt((u - P/2)**2 + (v - Q/2)**2)

#Definição do filtro HPF no domínio da frequência  para filtragem IDEAL, BUTTERWORTH e GAUSSIANA
def H_hpf(P, Q, D0, n, name='IHPF'):
    u = np.array(range(P))
    v = np.array(range(Q))

    coordinates = list(itertools.product(u,v))
    coordinates.sort(key=lambda x: x[1])

    if name == 'IHPF' :
        f_h = lambda t: 0 if D(t[0],t[1],P,Q) <= D0 else 1

    elif name == 'BHPF':
        f_h = lambda t: 1/(1 + ( D0/D(t[0],t[1],P,Q) )**(2*n) )

    elif name == 'GHPF':
        f_h = lambda t: 1 - np.exp(- ( ( D( t[0], t[1], P, Q)**2 )/(2*(D0**2)) ))

    hpf = np.array(list(map(f_h, coordinates)))
    hpf = np.reshape(hpf,(P,Q))

    return hpf

def bhpf(img):
    img_pad = img[:,:,1]
    img_dft = np.fft.fft2(img_pad)
    img_fshift = np.fft.fftshift(img_dft)
    img_spectrum = np.abs(img_fshift)
    img_spectrum_log = np.log(np.abs(img_fshift))

    H_matrix1 = H_hpf(img_pad.shape[0],img_pad.shape[1],D0=50,n=4, name='GHPF')

    mult_res1 = img_fshift*H_matrix1
    f_ishift1 = np.fft.ifftshift(mult_res1)
    inverse_mult1 = np.fft.ifft2(f_ishift1)

    # Definição de uma máscara de numeros 0
    mask_zeros = np.ma.masked_array(inverse_mult1.real, mask=(inverse_mult1.real<=0.0))
    limiar = np.ma.filled(mask_zeros,0.0)

    # Definição de uma máscara de números 1
    mask_ones = np.ma.masked_array(limiar.real, mask=(limiar.real>0.0))
    limiar = np.ma.filled(mask_ones,255.0)

    return limiar


def gray_segmentation(img):
    # Assuming the input image is already grayscale, no need for color conversions

    # Apply adaptive thresholding to the grayscale image
    image_gray_thresholded = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3
    )

    # Label connected components
    labels = measure.label(image_gray_thresholded, connectivity=2, background=0)
    mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')

    # Loop through each unique label
    for label in np.unique(labels):
        if label == 0:
            continue  # Ignore background

        # Create a mask for the current label
        label_mask = np.zeros(image_gray_thresholded.shape, dtype='uint8')
        label_mask[labels == label] = 255

        # Count the number of pixels in the label
        num_pixels = cv2.countNonZero(label_mask)

        # If the component is within the size range, add it to the final mask
        if 300 < num_pixels < 1500:
            mask = cv2.add(mask, label_mask)

    return mask