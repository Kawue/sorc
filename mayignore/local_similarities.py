import numpy as np
from scipy.ndimage.filters import generic_filter
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


def local_similarity_map(X, Y, size, measure):
    if measure not in ["pearson", "cosine"]:
        raise ValueError("measure parameter currently only support: 'pearson' or 'cosine'")

    def patch_similarity(patchX, patchY, measure):
        if measure == "pearson":
            sim = pearsonr(patchX.ravel(), patchY.ravel())[0]
            if np.isfinite(sim):
                return sim
            else:
                return 0
        elif measure == "cosine":
            sim  = 1 - cosine(patchX.ravel(), patchY.ravel())
            if np.isfinite(sim):
                return sim
            else:
                return 0
        else:
            raise ValueError("entropy_type has to be one of the following: 'pearson'.")
    
    if size % 2 == 0:
        raise ValueError("Window size must be odd.")
    s = size//2

    lsm = np.zeros(X.shape)
    X = np.pad(X, s, "constant", constant_values=0)
    Y = np.pad(Y, s, "constant", constant_values=0)
    for i in range(s, X.shape[0]-s):
        for j in range(s, X.shape[1]-s):
            patchX = X[i-s:i+s+1, j-s:j+s+1]
            patchY = Y[i-s:i+s+1, j-s:j+s+1]
            e = patch_similarity(patchX, patchY, measure)
            lsm[i-s,j-s] = e

    return lsm