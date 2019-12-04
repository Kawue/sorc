import sys
sys.path.append('.')
import numpy as np
from scipy.spatial.distance import hamming, dice, jaccard, rogerstanimoto,  kulsinski, sokalsneath, russellrao, yule
from helper.image_operators import binarize
from helper.filters import std_filter

def matching_distance(img1, img2, index_mask, method, weighted=False, weightmode=None, bias=None):
    if weightmode not in [None, "binary", "match", "mismatch"]:
        raise ValueError("weightmode has to be one of the following parameters: None, 'binary', 'match', 'mismatch'")

    if bias not in [None, "X", "Y"]:
        raise ValueError("weightmode has to be one of the following parameters: None, 'X', 'Y', 'mismatch'")


    methods = {
        "hamming": hamming,
        "dice": dice,
        "jaccard": jaccard,
        "rogerstanimoto": rogerstanimoto,
        "kulsinski": kulsinski,
        "sokalsneath": sokalsneath,
        "russellrao": russellrao,
        "yule": yule
    }

    # create binary versions of both images
    bimg1 = binarize(img1, mask=index_mask)
    bimg2 = binarize(img2, mask=index_mask)
    bimg12 = bimg1 * bimg2

    # create weighting combinations based on both images and (mis)matched areas
    biases = {
        None: (img1 + img2) / 2,
        "X": img1,
        "Y": img2
    }

    binary_biases = {
        None: bimg12,
        "X": bimg1,
        "Y": bimg2
    }

    weightmodes={
        None: biases[bias],
        "binary": binary_biases[bias] * biases[bias],
        "match": (bimg1*bimg2) * biases[bias],
        "mismatch": ((1-bimg1)*(1-bimg2)) * biases[bias]
    }

    # Return weighted or unweighted distance
    if weighted:
        return methods[method](bimg1.flatten(), bimg2.flatten(), weightmodes[weightmode].flatten())
    else:
        return methods[method](bimg1.flatten(), bimg2.flatten())



def matching_distance_map(img1, img2, index_mask, method, radius):
    methods = {
        "hamming": hamming,
        "dice": dice,
        "jaccard": jaccard,
        "rogerstanimoto": rogerstanimoto,
        "kulsinski": kulsinski,
        "sokalsneath": sokalsneath,
        "russellrao": russellrao,
        "yule": yule
    }

    # create binary versions of both images
    bimg1 = binarize(img1, mask=index_mask)
    bimg2 = binarize(img2, mask=index_mask)

    # Pad images to avoid border problems
    bimg1_padded = np.pad(bimg1, radius, "constant", constant_values=0)
    bimg2_padded = np.pad(bimg2, radius, "constant", constant_values=0)
    # Stack images for calculation purposes
    padstack = np.dstack((bimg1_padded, bimg2_padded))
    # Empty distance map
    dmap = np.zeros(padstack.shape[0:2])
    # Iterate over indices of the original array within the padded array
    for i in np.arange(padstack.shape[0])[:-2*radius]+radius:
        for j in np.arange(padstack.shape[1])[:-2*radius]+radius:
            # Original z layers are between nbrh and -nbrh, every other layer is padding.
            s_cube = padstack[i-radius:i+radius+1, j-radius:j+radius+1]
            # Caclulate intensity difference
            dist = methods[method](s_cube[:, :, 0].flatten(), s_cube[:, :, 1].flatten())
            # If both vectors are 0-vectors numpy results in nan.
            # My intuition says that even if both vectors are 0-vectors their sets are the same and the distance should be zero.
            if np.isnan(dist):
                if (s_cube[:, :, 0] == 0).all() and (s_cube[:, :, 1] == 0).all():
                    dist = 0
            # Set distances in distance map
            # -nbrh because the loop started with +nbrh
            dmap[i,j] = dist

    return dmap[radius:-radius, radius:-radius]



def homology_similarity_rr(X, Y, index_mask, reward_nosignal=True):
    mask_img = np.zeros_like(X)
    mask_img[index_mask] = 1

    # Calculating the Russell-Rao distance map with the images themselve provides a map of homogenous regions.
    Xmdm = matching_distance_map(X, X, index_mask=index_mask, method="russellrao", radius=2)
    Ymdm = matching_distance_map(Y, Y, index_mask=index_mask, method="russellrao", radius=2)
    Xmsm = (1 - Xmdm) * mask_img
    Ymsm = (1 - Ymdm) * mask_img

    if reward_nosignal:
        # Calculation of Hamming distance between the homogenous regions maps.
        # This rewards homogenous regions of signal and no signal with the same spatial position and punishes differences.
        msm_map = (1 - matching_distance_map(Xmsm, Ymsm, index_mask=index_mask, method="hamming", radius=2)) * mask_img
    else:
        # Russel-Rao evaluates only matching points vs all points. This approach does reward homogenous signal areas.
        msm_map = (1 - matching_distance_map(Xmsm, Ymsm, index_mask=index_mask, method="russellrao", radius=2)) * mask_img

    # Pooling. Alternative would be to calculate the Hamming distance on the whole images directly.
    msm_pooled = np.mean(msm_map[index_mask])

    return msm_pooled


# Very Similar alternative homology_similarity_rr
def homology_similarity_std(X, Y, index_mask, win_size=5, reward_nosignal=True):
    mask_img = np.zeros_like(X)
    mask_img[index_mask] = 1

    Xstd = std_filter(X, win_size, gaussian_weights=True)
    Ystd = std_filter(Y, win_size, gaussian_weights=True)
    std_msm = (1 - matching_distance_map(Xstd, Ystd, index_mask=index_mask, method="hamming", radius=2)) * mask_img
    std_msm_pooled = np.mean(std_msm[index_mask])
    return std_msm_pooled