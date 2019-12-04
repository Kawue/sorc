import numpy as np
import cv2
from scipy.stats import wasserstein_distance, energy_distance, ks_2samp


def image_histogram(img):
    if img.dtype == np.float:
        hist, bin_edges = histogram1d(img, bins=256, range=(0.0, 1.0))
    elif img.dtype == np.uint8:
        hist, bin_edges = histogram1d(img, bins=256, range=(0, 256))
    else:
        raise ValueError("Image has to be of type float or uint8")
    return hist, bin_edges


# Feature für das das Histogramm erstellt wird per parameter?
def histogram1d(data, bins=None, range=None):
    # Bins can be left as "auto"
    # set to one of the numpy variants ['fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']
    # or manually set by a binning function which returns an array where each entry describes a consecutive edge of the bins.
    if bins:
        if range:
            hist, bin_edges = np.histogram(data, bins=bins, range=range)
        else:
            hist, bin_edges = np.histogram(data, bins=bins)
    else:
        try:
            # Freedman Diaconis
            v25, v75 = np.percentile(data, [25, 75])
            fd = 2 * (v75 - v25) / (data.size ** (1 / 3))
            bins = int(np.ceil((data.max() - data.min())/fd))
            hist, bin_edges = np.histogram(data, bins=bins)
        except:
            # Sturges rule
            bins = int(np.log2(data.size) + 1)
            hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges


# Input x/y is the feature domain, in which the occurences described by the bins will be count. x will mostly be a pixel based domain, e.g. intensity value or gradient at each pixel. y can vary, e.g. neighborhood size, neighborhood intensities, etc.
# If you want to use this as image histogram 2d, use the bins of the image_histogram() method.
# Hist has a shape of (X,Y), meaning that the x-binning is along the vertical array axis and the y-binning is along the horizontal array axis.
def histogram2d(data_x, data_y, bins_x, bins_y):
    # Bins have to be set manually. Use a one of the helper functions to create them or provide the bins directly
    hist, xbin_edges, ybin_edges = np.histogram2d(data_x.ravel(), data_y.ravel(), bins=[bins_x, bins_y])
    return hist, xbin_edges, ybin_edges



# Methods from cv2 can be either: 
#   0: HISTCMP_CORREL (Correlation),
#   1: HISTCMP_CHISQR (Chi-Square),
#   2: HISTCMP_INTERSECT (Intersection),
#   3: HISTCMP_BHATTACHARYYA (synonym: HISTCMP_HELLINGER) (Hellinger distance), 
#   4: HISTCMP_CHISQR_ALT (alternative Chi-Square formulation, regularly used for texture comparison), 
#   5: HISTCMP_KL_DIV (Kullback-Leibler divergence) Note: Normalize the histograms so that their sum to 1 to achieve the same value as in scipy
#   6: Wasserstein Distance
#   7: Energy Distance
#   8: Kolmogorov-Smirnov Distance
#   Details: https://docs.opencv.org/3.2.0/d6/dc7/group__imgproc__hist.html#ga994f53817d621e2e4228fc646342d386

def hist_compare(hist1, hist2, method):
    hist1, hist2 = hist1.copy(), hist2.copy()
    if hist1.shape[0] != hist1.size:
        hist1 = hist1.ravel()

    if hist2.shape[0] != hist2.size:
        hist2 = hist2.ravel()

    if hist1.dtype != 'float32':
        hist1 = hist1.astype('float32')
    if hist2.dtype != 'float32':
        hist2 = hist2.astype('float32')
    
    if method in [0,1,2,3,4,5]:
        return cv2.compareHist(hist1, hist2, method)
    elif method == 6:
        return wasserstein_distance(hist1, hist2)
    elif method == 7:
        return energy_distance(hist1, hist2)
    elif method == 8:
        return ks_2samp(hist1, hist2)[0]
    else:
        raise ValueError("Method has to be an integer betwenn 0-7")