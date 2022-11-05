#Importar librerias 
import pydicom as dcm 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/Practica/Documents/Python/")
import mypdi_module as pdi

from skimage.filters import threshold_multiotsu

from scipy import signal
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_bregman

def perform_normalize(img):
  new_img= img.astype('float64')
  min_val = new_img.min()
  max_val = new_img.max()
  new_img = (new_img - min_val)/(max_val - min_val)
  return new_img

def convert_integer(img, max_val):
  new_img = perform_normalize(img)
  new_img = np.around(max_val * new_img)
  new_img = new_img.astype('uint16')
  return new_img

def perform_histogram_cdf(img):
  idx = img != 0   #Indexación de los elementos diferentes de 0
  imgVec = img[idx] # Vector con intensidades diferentes de 0
  max_val = imgVec.max() + 1
  hist, _ = np.histogram(imgVec, max_val, [0, max_val])
  cdf = hist.cumsum()

  #normalize cdf
  cdf_min = cdf.min()
  cdf_max = cdf.max()
  cdf_m = np.ma.masked_equal(cdf, 0)
  cdf_m = max_val * ((cdf_m - cdf_min)/(cdf_max - cdf_min))
  cdf = np.ma.filled(cdf_m,0).astype('uint16')
  return hist, cdf

def perform_histogram_transformation(img, cdf):
  M, N = img.shape
  idx = img != 0   #Indexación de los elementos diferentes de 0
  img_vec = img[idx] # Vector con intensidades diferentes de 0

  equ = cdf[img_vec]

  img_equ = np.zeros((M,N), dtype='uint16')
  img_equ[idx] = equ

  return img_equ

objdcm = dcm.dcmread('000043.dcm')
img = objdcm.pixel_array
max_val = img.max()

img = perform_normalize(img)
sig_est = np.mean(estimate_sigma(img))
patch_w = dict(patch_size = 5,
               patch_distance = 12)
#non-local mean
img_nlm = denoise_nl_means(img, h = 7*sig_est, fast_mode = False,
                           **patch_w)

plt.imsave('reduRuido1.png', img_nlm, cmap = 'gray')

# histogram equalization
img_nlm = convert_integer(img_nlm, max_val)
_, cdf = perform_histogram_cdf(img_nlm)
img_nlm = perform_histogram_transformation(img_nlm, cdf)

plt.imsave('reduRuido2.png', img_nlm, cmap = 'gray')

plt.hist(img_nlm)
plt.title('Histograma Segmentacion 4-Nov')
plt.savefig('Histograma_SEG.png')


# SEGMENTACIÓN
##Normalizar imagen entre 0 y 255
img8 = pdi.normImage(img_nlm)
img8 = np.around(255*img8).astype('uint8')

##Calculando umbrales, calcula los óptimos como lo hicimos manualmente 
thresh = threshold_multiotsu(img8, classes = 3)

#Umbralizar la imagen
regions = np.digitize(img8, bins = thresh)
seg1 = (regions == 0).astype('int')
seg2 = (regions == 1).astype('int')
seg3 = (regions == 2).astype('int')

#Graficamos el resultado
fig, axs = plt.subplots(2, 3, figsize =(10,6))
axs[0,0].imshow(img8, cmap = 'gray')
axs[0,0].set_title('Original')
axs[0,0].axis('off')

axs[0,1].hist(img8.ravel(), bins = 255)
axs[0,1].set_title('Histograma con umbrales')
for t in thresh:
    axs[0,1].axvline(t, color = 'r')

axs[0,2].imshow(regions, cmap= 'jet')
axs[0,2].set_title('Imagen segmentada')
axs[0,2].axis('off')

axs[1,0].imshow(seg1, cmap= 'gray')
axs[1,0].set_title('Segemento 1')
axs[1,0].axis('off')

axs[1,1].imshow(seg2, cmap= 'gray')
axs[1,1].set_title('Segemento 2')
axs[1,1].axis('off')

axs[1,2].imshow(seg3, cmap= 'gray')
axs[1,2].set_title('Segemento 3')
axs[1,2].axis('off')

plt.show()