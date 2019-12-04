import sys
sys.path.append('.')
import numpy as np
from helper.histogram_helper import histogram1d, histogram2d, image_histogram

def log(x, base=None):
    if base:
        return np.log(x) / np.log(base)
    else:
        return np.log(x)

def validate_distr(distr):
    if np.around(np.sum(distr), 5) != 1:
        return distr/distr.sum()
    else:
        return distr


def prepare_histograms(input1, input2, bins=False):
    Hx, xbins = histogram1d(input1)
    Hy, ybins = histogram1d(input2)
    Hxy, xxybins , yxybins = histogram2d(input1, input2, xbins, ybins)
    if bins:
        return Hx, xbins, Hy, ybins, Hxy, xxybins, yxybins
    else:
        return Hx, Hy, Hxy


def prepare_image_histograms(image1, image2, bins=False):
    Hx, xbins = image_histogram(image1)
    Hy, ybins = image_histogram(image2)
    Hxy, xxybins , yxybins = histogram2d(image1, image2, xbins, ybins)
    if bins:
        return Hx, xbins, Hy, ybins, Hxy, xxybins, yxybins
    else:
        return Hx, Hy, Hxy