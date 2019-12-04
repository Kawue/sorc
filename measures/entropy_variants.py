import sys
sys.path.append('.')
import numpy as np
from scipy.ndimage.filters import generic_filter
from scipy.stats import wasserstein_distance as sp_wd
from helper.entropy_helper import validate_distr, log, prepare_image_histograms
from helper.histogram_helper import image_histogram

##### Entropy functions #####
# Maximal Entropy is given by log(N), where N is the number of samples
def entropy(Hx, base=None):
    Hx = validate_distr(Hx)
    return -1  * np.nansum(Hx * log(Hx, base))

def joint_entropy(Hxy, base=None):
    Hxy = validate_distr(Hxy)
    return -1 * np.nansum(Hxy * log(Hxy, base))

# Conditional Entropy of Hy with respect to Hx. Turn input for Hx with respect to Hy.
def conditional_entropy(Hy, Hx, Hxy, base=None):
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    Hxy = validate_distr(Hxy)
    return joint_entropy(Hxy, base) - entropy(Hx, base)

# q can be any real number
def tsallis_entropy(Hx, q):
    Hx = validate_distr(Hx)
    return (1 - (np.sum(np.power(Hx, q)))) / (q-1)

# a has to be greater or equal to zero and unequal to one
# a = 2 results in collision entropy
def renyi_entropy(Hx, a, base=None):
    if a < 0 or a == 1:
        raise ValueError("a has to be greater or equal to zero and unequal to one.")
    Hx = validate_distr(Hx)
    return log(np.sum(np.power(Hx, a)), base)/(1-a)

def collision_entropy(Hx, base=None):
    Hx = validate_distr(Hx)
    return renyi_entropy(Hx, 2, base)





##### Entropy based measures #####
def mutual_information(Hx, Hy, Hxy):
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    Hxy = validate_distr(Hxy)
    return entropy(Hx) + entropy(Hy) - joint_entropy(Hxy)


# Kullback-Leibler Divergence (aka Information Gain) of Hx with respect to Hy.
# Symmetrie is achieved by adding both directions.
# In case if only zeros after division the term will be inf, resulting in empty values if plottet.
def kullback_leiber_divergence(Hx, Hy, symmetric, base=None):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    if symmetric:
        Hx_div_Hy = np.divide(Hx, Hy, out=np.zeros_like(Hx), where=Hy!=0)
        Hy_div_Hx = np.divide(Hy, Hx, out=np.zeros_like(Hy), where=Hx!=0)
        logterm_xy = log(Hx_div_Hy, base)
        logterm_xy[np.where(np.invert(np.isfinite(logterm_xy)))] = 0
        logterm_yx = log(Hy_div_Hx, base)
        logterm_yx[np.where(np.invert(np.isfinite(logterm_yx)))] = 0
        kl_xy = np.sum(Hx * logterm_xy)
        kl_yx = np.sum(Hy * logterm_yx)
        return kl_xy + kl_yx
    else:
        Hx_div_Hy = np.divide(Hx, Hy, out=np.zeros_like(Hx), where=Hy!=0)
        logterm_xy = log(Hx_div_Hy, base)
        logterm_xy[np.where(np.invert(np.isfinite(logterm_xy)))] = 0
        return np.sum(Hx * logterm_xy)


# Jensen-Shannon Divergence is based on Kullback-Leibler Divergence and already symmetric.
def jensen_shannon_divergence(Hx, Hy, base=None):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    m = 0.5 * (Hx + Hy)
    kl_xm = kullback_leiber_divergence(Hx, m, base, False)
    kl_ym = kullback_leiber_divergence(Hy, m, base, False)
    return 0.5 * (kl_xm + kl_ym)


def jensen_shannon_distance(Hx, Hy, base=None):
    return np.sqrt(jensen_shannon_divergence(Hx, Hy, base))


# In case if only zeros after division the term will be inf, resulting in empty values if plottet.
def renyi_divergence(Hx, Hy, a, symmetric, base=None):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    if a <= 0 or a == 1:
        raise ValueError("a has to be greater than zero and unequal to one.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    if symmetric:
        Hx_div_Hy = np.divide(np.power(Hx, a), np.power(Hy, a-1), out=np.zeros_like(Hx), where=Hy != 0)
        Hy_div_Hx = np.divide(np.power(Hy, a), np.power(Hx, a-1), out=np.zeros_like(Hy), where=Hx != 0)
        r_xy = log(np.nansum(Hx_div_Hy), base) / (a-1)
        r_yx = log(np.nansum(Hy_div_Hx), base) / (a-1)
        return r_xy + r_yx
    else:
        Hx_div_Hy = np.divide(np.power(Hx, a), np.power(Hy, a-1), out=np.zeros_like(Hx), where=Hy != 0)
        return log(np.nansum(Hx_div_Hy), base) / (a-1)


def bhattacharyya_coefficient(Hx, Hy):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    return np.sqrt(Hx*Hy).sum()


def bhattacharyya_distance(Hx, Hy, base=None):
    bc = bhattacharyya_coefficient(Hx, Hy)
    bd = -log(bc, base)
    return bd


# H=sqrt(1-BC) and BC=1-H**2 -> For alternative definitions
def hellinger_distance(Hx, Hy):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    hd = np.sqrt(((np.sqrt(Hx) - np.sqrt(Hy))**2).sum()) / np.sqrt(2)
    return hd


def wasserstein_distance(Hx, Hy):
    if Hx.size != Hy.size:
        raise ValueError("Hx and Hy must have same length.")
    Hx = validate_distr(Hx)
    Hy = validate_distr(Hy)
    wd = sp_wd(Hx, Hy)
    return wd





##### Local Entropy Maps #####
# Size describes the full width and height of the window, i.e. window.shape == (size,size).
def local_entropy_map(X, size=5, entropy_type="shannon", base=None, param=None):
    def local_entropy(values, entropy_type, base, param):
        values = values.copy()
        Hx, xbins = image_histogram(values)
        if entropy_type == "shannon":
            return entropy(Hx, base)
        elif entropy_type == "tsallis":
            return tsallis_entropy(Hx, param)
        elif entropy_type == "renyi":
            return renyi_entropy(Hx, param, base)
        elif entropy_type == "collision":
            return collision_entropy(Hx, base)
        else:
            raise ValueError("Entropy needs to be shannon, tsallis, renyi or collision")
    
    if size % 2 == 0:
        raise ValueError("Size needs to be uneven")

    lemap = generic_filter(X, local_entropy, size=size, extra_arguments=(entropy_type, base, param, ))
    return lemap


def local_entropy_map2(X, Y, size=5, entropy_type="jointentropy", base=None, symmetric=True, a=2):
    def local_entropy2(patchX, patchY, entropy_type, base=base, symmetric=symmetric, a=a):
        patchX, patchY = patchX.copy(), patchY.copy()
        Hx, Hy, Hxy = prepare_image_histograms(patchX, patchY, bins=False)
        if entropy_type == "jointentropy":
            return joint_entropy(Hxy, base=base)
        elif entropy_type == "conditionalentropy":
            return conditional_entropy(Hx, Hy, Hxy, base=base)
        elif entropy_type == "mutualinformation":
            return mutual_information(Hx, Hy, Hxy)
        elif entropy_type == "kullbackleiblerdivergence":
            return kullback_leiber_divergence(Hx, Hy, symmetric=symmetric, base=base)
        elif entropy_type == "jensenshannondivergence":
            return jensen_shannon_divergence(Hx, Hy, base=base)
        elif entropy_type == "jensenshannondistance":
            return jensen_shannon_distance(Hx, Hy, base=base)
        elif entropy_type == "renyidivergence":
            return renyi_divergence(Hx, Hy, a=a, symmetric=symmetric, base=base)
        elif entropy_type == "bhattacharyyacoefficient":
            return bhattacharyya_coefficient(Hx, Hy)
        elif entropy_type == "bhattacharyyadistance":
            return bhattacharyya_distance(Hx, Hy, base)
        elif entropy_type == "hellingerdistance":
            return hellinger_distance(Hx, Hy)
        elif entropy_type == "wassersteindistance":
            return wasserstein_distance(Hx, Hy)
        else:
            raise ValueError("entropy_type has to be one of the following: 'jointentropy', 'mutualinformation', 'kullbackleiblerdivergence', 'jensenshannondivergence', 'jensenshannondistance', 'renyidivergence', 'bhattacharyyacoefficient', 'bhattacharyyadistance', 'hellingerdistance'")
    
    if size % 2 == 0:
        raise ValueError("Size needs to be uneven")
    
    s = size//2
    lemap = np.zeros(X.shape)
    X = np.pad(X, s, "constant", constant_values=0)
    Y = np.pad(Y, s, "constant", constant_values=0)
    for i in range(s, X.shape[0]-s):
        for j in range(s, X.shape[1]-s):
            patchX = X[i-s:i+s+1, j-s:j+s+1]
            patchY = Y[i-s:i+s+1, j-s:j+s+1]
            e = local_entropy2(patchX, patchY, entropy_type, base, symmetric, a)
            lemap[i-s,j-s] = e
    
    return lemap