import os
import numpy as np
from imageio import imread
from skimage import img_as_float

def read_images(dirpath, padding):
    imgs = []
    labels = []
    for filename in os.listdir(dirpath):
        if len(filename.split(".p")) > 1:
            img = imread(os.path.join(dirpath, filename), pilmode="L")
            img = img_as_float(img)
            img = np.pad(img, padding, "constant", constant_values=0)
            imgs.append(img)
            filename = filename.split(".p")[0]
            labels.append(filename)
    return np.array(imgs), np.array(labels)


# Function to consider only the measured area of the image and set every other value to 0
# Artificial black background can corrupt some of the measures    
# Padding avoids border effects
def index_mask(dframe, padding, x_pixel_identifier="grid_x", y_pixel_identifier="grid_y"):
    try:
        x = (dframe.index.get_level_values(x_pixel_identifier) + padding).astype(int)
        y = (dframe.index.get_level_values(y_pixel_identifier) + padding).astype(int)
        img = np.zeros((y.max() + 1 + padding, x.max() + 1 + padding))
        img[(y,x)] = 1
        indices = np.where(img==1)
        return indices, img
    except:
        print("The given dataframe provides the following index names:")
        print(dframe.index.names)
        print("Please select the correct identifier defining x and y pixels.")