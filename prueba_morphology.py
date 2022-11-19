import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm

import mypdi_module as pdi

from skimage.morphology import erosion, dilation
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_multiotsu

objdcm = dcm.dcmread('000197.dcm')
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

#full holes
fullHoles = binary_fill_holes(seg3)

filtro = np.array([[1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1] ])

#Dilatación 
img_Erosion = erosion(fullHoles, filtro)
img_Dilation = dilation(img_Erosion, filtro)


fig, axs = plt.subplots(2, 3, figsize =(10,6))
axs[0,0].imshow(img8, cmap = 'gray')
axs[0,0].set_title('Original')
axs[0,0].axis('off')

axs[0,1].imshow(seg3, cmap = 'gray')
axs[0,1].set_title('Segmento')
axs[0,0].axis('off')

axs[0,2].imshow(fullHoles, cmap= 'gray')
axs[0,2].set_title('Full Holes')
axs[0,2].axis('off')

axs[1,0].imshow(img_Dilation, cmap= 'gray')
axs[1,0].set_title('Dilatación')
axs[1,0].axis('off')

axs[1,1].imshow(img_Erosion, cmap= 'gray')
axs[1,1].set_title('Erosion')
axs[1,1].axis('off')

plt.show()

plt.imsave('org_Morpho.png', img8, cmap = 'gray')
plt.imsave('seg_Morpho.png', seg3, cmap = 'gray')
plt.imsave('full_Holes.png', fullHoles, cmap = 'gray')
plt.imsave('dilatacion.png', img_Dilation, cmap = 'gray')
plt.imsave('erosion.png', img_Erosion, cmap = 'gray')
