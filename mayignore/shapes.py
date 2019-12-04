import numpy as np
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

### Shape images has to contain only a single shape.

# Paper: Automatic Detection and Recognize Different Shapes in an Image
def shape_factor(shape_image):
    shape_image = shape_image.copy()

    label_img, num = label(shape_image, connectivity=1, return_num=True)
    props = regionprops(label_img)
    area = props[0].area

    convex_image = convex_hull_image(shape_image, offset_coordinates=True)
    max_diameter = feret_diameter(convex_image, "max")

    return area / max_diameter**2


# feret max diameter
def feret_diameter(convex_image, min_max):
    coordinates = np.vstack(find_contours(convex_image, level=0, fully_connected="high")).astype(np.int)
    outline_image = np.zeros((convex_image.shape[0] + 10, convex_image.shape[1] + 10))
    outline_image[coordinates[:,0], coordinates[:,1]] = 1
    distances = pdist(coordinates, 'euclidean')
    if min_max == "max":
        return np.max(distances)
    elif min_max == "min":
        return np.min(distances)
    else:
        raise ValueError("min_max needs to be 'min' or 'max'")


def shape_extraction(image):
    pass


# Paper: A shape factor to assess the shape of particles using image analysis
def shape_factor2(shape_image):
    pass