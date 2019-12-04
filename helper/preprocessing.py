import sys
sys.path.append('.')
import numpy as np
import skimage.filters as skif
from skimage import restoration as res
from scipy.ndimage import filters
from helper.scalespace import scale_space_gauss

def pxpxvariationreduction(X, index_mask, mask, method):
    Xs = X.copy()
    if method == "otsus":
        t = skif.threshold_otsu(Xs[index_mask], nbins=256)
        Xs[Xs < t] = 0
        return Xs
    elif method == "isodata": # embryo1 shows very similar to otsus
        t = skif.threshold_isodata(Xs[index_mask], nbins=256)
        Xs[Xs < t] = 0
        return Xs
    elif method == "li":
        t = skif.threshold_li(Xs[index_mask])
        Xs[Xs < t] = 0
        return Xs
    elif method == "chambolle":
        #TV-Chambolle denoising
        Xs = res.denoise_tv_chambolle(Xs, weight=0.08,eps=0.000002)
        return Xs
    elif method == "bregman-iso":
        #TV-Bregman denoising
        Xs = res.denoise_tv_bregman(Xs, weight=6, isotropic=True)
        return Xs
    elif method == "bregman-aniso":
        #TV-Bregman anisotropic denoising
        Xs = res.denoise_tv_bregman(Xs, weight=12, isotropic=False)
        return Xs
    elif method == "median": # too harsh
        #Median filter
        Xs = filters.median_filter(Xs, size = (5,5))
        return Xs
    elif method == "scalespace":
        Xs = scale_space_gauss(Xs, downscale=1, steps=1, sigma=0.8)
        return Xs[1]
    elif method == "bilateral":
        Xs = res.denoise_bilateral(Xs, sigma_color=2*np.std(X), sigma_spatial=2, multichannel=False)
        return Xs
    else:
        raise ValueError("Wrong Value for method in denoiser()")


def preprocess_images(imgs, index_mask, mask_img):
    pp_imgs = []
    for X in imgs:
        #X = X / X.sum()
        Xs = pxpxvariationreduction(X, index_mask, mask_img, "otsus")
        Xs = pxpxvariationreduction(Xs, index_mask, mask_img, "scalespace")
        #winsorize(Xs, limits=(0, 0.01), inplace=True)
        #Xs = Xs*mask_img
        pp_imgs.append(Xs)
    return np.array(pp_imgs)
