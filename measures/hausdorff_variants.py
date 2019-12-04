import numpy as np
from scipy.spatial.distance import directed_hausdorff
# https://github.com/mavillan/py-hausdorff
from hausdorff import hausdorff

def hausdorff_distance(img1, img2, method, symmetric=True):
    if symmetric:
        return max(hausdorff(img1, img2, distance=method), hausdorff(img2, img1, distance=method))
    else:
        return hausdorff(img1, img2, distance=method)