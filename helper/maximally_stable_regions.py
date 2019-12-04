import numpy as np
from skimage.measure import regionprops, label

# Currently Restricted to uInt8 Values!

# If the Image has a large background region, e.g. ims barley seed, the max_area parameter must be respectively small.
# E.g. if the background is 50% of the image and max_area is > 0.5 the whole foreground will be included, since it is smaller than the max_area parameter.

# Own implementation to find maximally stable regions
# Original Paper: "Robust wide-baseline stereo from maximally stable extremal regions"
def calc_nested_regions(img, delta, min_area, max_area):
    delta = delta
    if isinstance(min_area, int):
        min_area = min_area
    elif isinstance(min_area, float):
        min_area = int(img.size * min_area)
    else:
        raise ValueError("min_area needs to be of type int or float")

    if isinstance(max_area, int):
        max_area = max_area
    elif isinstance(max_area, float):
        max_area = int(img.size * max_area)
    else:
        raise ValueError("min_area needs to be of type int or float")

    img_dict = {}
    nested_regions = { -np.inf: [np.zeros(img.shape)] }
    counter = 0
    
    img_crop = np.full(img.shape, 255)

    # Cut away low intensities first
    for i in range(0,256,delta):
        img_crop[np.where(img == i)] = 0
        image_copy = np.zeros(img.shape) + img_crop
        labels, num = label(image_copy, connectivity=1, return_num=True)
        img_dict[i] = {
            "image": image_copy,
            "regions": labels,
            "number": num
        }
    
    for i in range(0,256,delta):
        update_dict = {}
        for region in regionprops(img_dict[i]["regions"]):
            if min_area < region.area < max_area:
                t = np.zeros(img.shape)
                t[region.coords[:,0], region.coords[:,1]] = 1
                for key in nested_regions:
                    if ((nested_regions[key][-1] - t) > -1).all():
                        # Needed because the update dict begins empty at each loop.
                        try:
                            update_dict[key].append(t)
                        except:
                            update_dict[key] = [t]
                        break
                else:
                    update_dict[counter] = [t]
                    counter += 1

        # Check if update is needed.
        # There is no uopdate needed when the region area is outside of min/max area, which results in an empty update dict.
        if update_dict:
            key_adjust = max(max(update_dict.keys()),max(nested_regions.keys())) + 1
            # Update
            for key in update_dict:
                if key in nested_regions:
                    if len(update_dict[key]) > 1:
                        for reg in update_dict[key]:
                            update_region = nested_regions[key][:]
                            update_region.append(reg)
                            nested_regions[key_adjust] = update_region
                            key_adjust += 1
                        del nested_regions[key]
                    else:
                        nested_regions[key].extend(update_dict[key])
                else:
                    nested_regions[key] = update_dict[key][:]
            counter = key_adjust + 1
    else:
        del nested_regions[-np.inf]
    return nested_regions


def calc_stability(bigger_region, current_region, smaller_region):
    return (bigger_region.sum() - smaller_region.sum()) / current_region.sum()


# sequence_min: the length of a sequence of regions to be considered as stable. Minimum is three.
# delta: step size of color increase.
# min_area: minimum area for a regions to be considered as stable.
# max_area: maximum area for a regions to be considered as stable.
def calc_mser(img, sequence_min, delta, min_area, max_area, all_regions=False):
    if img.dtype != np.uint8:
        raise ValueError("This method currently needs uint8 format!")

    # The sequence length needs to be at least 3 to evaluate the stability.
    if sequence_min < 3:
        raise ValueError("sequence_min must be at least 3.")
    else:
        sequence_min = sequence_min
    stability_dict = {}
    nested_regions = calc_nested_regions(img, delta, min_area, max_area)

    for key, regions in nested_regions.items():
        if len(regions) >= sequence_min:
            stabilities = []
            # Regions becoming smaller as the list goes on.
            # The reverse case is true in the original MSER paper.
            for idx, _ in enumerate(regions[1:-1], start=1):
                stability = calc_stability(regions[idx-1], regions[idx], regions[idx+1])
                stabilities.append(stability)
            stability_dict[key] = stabilities
    
    stable_regions = []
    for key, value in stability_dict.items():
        idx = np.argmin(value)
        stable_regions.append(nested_regions[key][idx])
        
    stable_regions = np.array(stable_regions)
    #TODO: Two or more series can have the same stable regions before they divided in later steps.
    if len(stable_regions) > 0:        
        # Sum all stable regions to combine information and visualize stability intensity.
        mser_img = np.sum(stable_regions, axis=0)
        if all_regions:
            return mser_img, stable_regions
        else:
            return mser_img
    else:
        print("ATTENTION! There are no stable regions in this image. There ist most likely not enough color or intensity information!")
        return np.zeros(img.shape), []