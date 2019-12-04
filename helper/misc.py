import numpy as np
from scipy.ndimage.morphology import binary_dilation

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('4th Argument expects Boolean.')

# Scale values from [0,1] to a target range of [a,b]
def scale_values(x,a,b):
    y = np.around(x,9)
    if (y < 0).any() or (y > 1).any():
        raise ValueError("Wrong Domain, not [0,1]")
    ymin = 0
    ymax = 1
    return (b-a) * ((y-ymin)/(ymax-ymin)) + a


# Rescale similarity or distance matrix into a new range, default is [0,1].
def rescale_datamatrix(dmatrix, new_min=0, new_max=1):
    mmax = np.amax(dmatrix)
    mmin = np.amin(dmatrix)
    scaled_dmatrix = (new_max-new_min) * ((dmatrix-mmin) / (mmax-mmin)) + new_min
    return scaled_dmatrix


def normalization(X, equalize_method):
    if equalize_method:
        if equalize_method == "all":
            return (X / X.sum())
        elif equalize_method == "solo":
            return ((X - np.amin(X)) / (np.amax(X) - np.amin(X)))
        else:
            raise ValueError("\n Method has to be either 'all' or 'solo'. \n"
                "   'all'  equalizes the image such that all intensities sum up to one, meaning that the total intensity of both normalized images are equal. \n "
                "   'solo' equalizes the intensity range of the image to [0,1], meaning that the total intensity of both normalized images will be different, but their minima will be 0 and their maxima will be 1.")
    else:
        print("ATTENTION: No normalization was done!")
        return X


# Calculate thr truncated value needed to achiece a specific Window Size with a specific Sigma in the scipy gaussian filter
# w: Windows Size
# s: Sigma
def set_window_size(w,s):
    t = (((w-1)/2)-0.5)/s
    return t


def extend_index_mask(mask_img, radius):
    mask_img_extended = binary_dilation(mask_img, iterations=radius)
    index_mask_extended = np.where(mask_img_extended == 1)
    return index_mask_extended, mask_img_extended