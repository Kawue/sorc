from skimage.feature import hog as sim_hog

# Orientations in the skimage hog implementation are based on 180°
def hog(img, orientations=9, pixels_per_cell=(9,9), cells_per_block=(3,3), block_norm="L2-Hys", visualize=True, return_map=True, feature_vector=True):
    hog_descriptor, hog_img = sim_hog(img, 
        orientations = orientations, 
        pixels_per_cell = pixels_per_cell, 
        cells_per_block = cells_per_block, 
        block_norm = block_norm, 
        visualize = visualize,
        transform_sqrt = False,
        feature_vector = feature_vector)
        
    if return_map:
        return hog_descriptor, hog_img
    else:
        return hog_descriptor