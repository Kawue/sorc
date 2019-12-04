import numpy as np
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from skimage.filters import gaussian


def preprocess_scalespace(imgs, method, steps):
    scalespace_list = [[] for i in range(steps)]
    for img in imgs:
        if method == "gauss":
            img_space = scale_space_gauss(img, downscale=1, steps=steps-1, sigma=0.8)
        elif method == "dog":
            img_space = scale_space_dog(img, downscale=1, steps=steps-1, sigma=0.8)
        else:
            raise ValueError("Method has to be 'gauss' or 'dog'.")
        # Put the first scale space image into the first sublist, second into second, etc. for each images.
        # This ends up with a structure of #steps x #images x height x width
        for idx, img in enumerate(img_space):
            scalespace_list[idx].append(img)
    return np.array(scalespace_list)


# Image Pyramid Scale Space with gaussian smoothing
# For Scale Space without downsampling use the gaussian filter repeatedly
# pyramid_gaussian
def scale_space_gauss(img, downscale, sigma=None, steps=None):
    if downscale < 1:
        raise ValueError("Downscale has to be greater or equal to one.")

    if downscale > 1:
        if steps or sigma:
            raise ValueError("For Downscale > 1, sigma and steps are determined by the downscaling parameter.")
        return list(pyramid_gaussian(img, downscale=downscale))
    else:
        if not steps or not sigma:
            raise ValueError("For Downscale = 1 sigma and the number of smoothing steps need to be determined. We propose sigma 1.")
        # To stay consistent with the skimage definitition -> Too Weak
        #s = 2 * downscale / 6.0
        s = sigma
        # Window size is not given currently to stay consistent with skimage
        # 4 is the default for truncated in ndimage
        #print("Gaussian Scale Space Window size: " + str(2*int(4*s+0.5)+1))

        img_layers = []

        layer_img = img
        img_layers.append(layer_img)
        
        for i in range(steps):
            layer_img = gaussian(layer_img, sigma=s)
            img_layers.append(layer_img)

        return img_layers



# Each layer contains the difference between the downsampled and the downsampled smoothed image
# Extracts the signal between two gaussian filter steps
# For Difference of gaussian without downsampling use gaussian filter repeatedly and build the difference between each consecutive use
# pyramid_laplacian
def scale_space_dog(img, downscale, sigma=1, steps=None):
    if downscale < 1:
        raise ValueError("Downscale has to be greater or equal to one.")
        
    if downscale > 1:
        if steps or sigma:
            raise ValueError("For Downscale > 1, sigma and steps are determined by the downscaling parameter.")
        return list(pyramid_laplacian(img, downscale=downscale))
    else:
        if not steps:
            raise ValueError("For Downscale = 1 sigma and the number of smoothing steps need to be determined.")
        # To stay consistent with the skimage definitition -> Too Weak
        #s = 2 * downscale / 6.0
        s = sigma
        # Window size is not given currently to stay consistent with skimage
        # 4 is the default for truncated in ndimage
        #print("Laplacian Scale Space Window size: " + str(2*int(4*s+0.5)+1))

        img_layers = []

        layer_img = img
        img_layers.append(layer_img)
        
        for i in range(steps):
            prev_layer_img = layer_img
            layer_img = gaussian(prev_layer_img, sigma=s)
            img_layers.append(prev_layer_img - layer_img)

        return img_layers