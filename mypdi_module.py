import numpy as np


def normImage(img, val):
    newimg = img.copy()
    newimg = newimg.astype('float64')
    maxval = newimg.max()
    minval = newimg.min()
    newimg = (newimg - minval)/(maxval - minval)
    newimg = np.around(val*newimg).astype('uint16')
    return newimg


def perform_hist_equalizer(img):
    idx = img != 0
    imgvec = img[idx]
    maxm = img.max()

    # image histogram
    hist, _ = np.histogram(imgvec, maxm+1, [0, maxm+1])
    cdf = hist.cumsum()
    maxcdf = cdf.max()
    mincdf = cdf.min()
    cdf_m = maxm*((cdf - mincdf)/(maxcdf - mincdf))
    cdf = np.ma.filled(cdf, 0).astype('uint16')

    return hist, cdf


def performHistTrans(img, lut):
    # Dimensions
    M, N = img.shape
    idx = img != 0
    imgvec = img[idx]
    equ = lut[imgvec]
    imgequ = np.zeros((M, N), dtype='uint16')
    imgequ[idx] = equ

    return imgequ
