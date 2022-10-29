import numpy as np
import matplotlib.pyplot as plt 
import pydicom as dcm

import mypdi_module as pdi

from skimage.filters import threshold_multiotsu

objdcm = dcm.dcmread('000194.dcm')
img = objdcm.pixel_array

##Normalizar imagen entre 0 y 255
img8 = pdi.normImage(img)
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

plt.hist(img8.ravel(), bins = 255)
plt.title('Histograma con umbrales')
for t in thresh:
    plt.axvline(t, color = 'r')
plt.savefig('Histograma_fig2.png')

plt.imsave('segmentada_fig2.png', regions, cmap = 'gray')
plt.imsave('seg1_fig2.png', seg1, cmap = 'gray')
plt.imsave('seg2_fig2.png', seg2, cmap = 'gray')
plt.imsave('seg3_fig2.png', seg3, cmap = 'gray')