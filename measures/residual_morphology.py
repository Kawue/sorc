import sys
sys.path.append('.')
import numpy as np
from helper.image_operators import morphology_filter, binarize, residual_image

#Resource: An image similarity measure method based on mathematical morphology
def residual_morphology_similarity(img1, img2, selem, m, n=None, return_map=False):
    # Calculate residual image
    res_img = residual_image(img1, img2)
    
    if (res_img == 0).all():
        if return_map:
            return 1, np.zeros(img1.shape)
        else:
            return 1

    # Binarize image 1, image 2 and residual image
    img1 = binarize(img1)
    img2 = binarize(img2)

    # Residual image has strong signal where only one of both images has a strong signal
    res_img = binarize(res_img)

    # Open the residual image
    res_img = morphology_filter(res_img, selem, False, m, n)

    # The bigger the signal in the residual image, the smaller distance value and the higher the similarity value
    s = 1 - res_img.sum() / np.sqrt(img1.sum() * img2.sum())

    if return_map:
        return s, res_img
    else:
        return s