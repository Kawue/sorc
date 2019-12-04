import sys
sys.path.append('.')
import numpy as np
from helper.gradient_helper import gradient_map, magnitude_map, direction_map
from scipy.ndimage.filters import generic_filter

# Reference: Image retrieval based on shape similarity by edge orientation autocorrelogram.
# Returns an autocorrelogramm as 2d histogramm. To compare two images, compare those histograms.
def eoac(img, sample_area=None, t=None):
    if not sample_area:
        sample_area = np.ones(img.shape)

    dy,dx = gradient_map(img)
    # Extract "prominent" edges. Threshold taken from reference paper.
    mag = magnitude_map(dy, dx)
    if not t:
        t = np.amax(img) * 0.1
    prom_idx = np.where(mag<t)
    deg = direction_map(dy, dx)
    # Make no difference between positive and negative angles to stay according to the reference paper.
    deg = np.abs(deg)
    deg[prom_idx] = 0

    deg_bin_edges = list(range(0,181,5))
    deg_bin_edges[-1] = 181
    deg_bins = list(zip(deg_bin_edges[:-1], deg_bin_edges[1:]))

    nbr_dist = range(1,8,2)

    # Each bin equals 5 degree
    # neighborhood dict of 5° bin dicts
    eoac_dict = {i: {j: 0 for j in deg_bins} for i in nbr_dist}

    for d, bins_dict in eoac_dict.items():
        footprint = np.zeros((d*2+1,d*2+1))
        # This is because of the special footprint format.
        midpoint = int(4*d)
        footprint[0,:] = footprint[-1,:] = footprint[:,0] = footprint[:,-1] = footprint[d,d] = 1
        generic_filter(deg, fill_bin_dict, footprint=footprint, mode="constant", cval=0, extra_arguments=(bins_dict, midpoint))

    # Two triple for loops, this is an efficiency nightmare and should be improved some day!
    # Generate histogram based on constructed dict
    nbr_info = []
    for k, i in eoac_dict.items():
        for kk, ii in i.items():
            for j in range(ii):
                nbr_info.append(k)

    deg_info = []
    for k, i in eoac_dict.items():
        for kk, ii in i.items():
           for j in range(ii):
                deg_info.append(kk[0] + 1)

    # create bin edges, where the original neighbors lie exactly in the middle of the bin
    nbr_bin_edges = [0]
    nbr_bin_edges.extend([(x[0] + x[1]) / 2 for x in list(zip(nbr_dist[:-1], nbr_dist[1:]))])
    nbr_bin_edges.append(nbr_dist[-1] + 1)

    # create autocorrelogram as 2d histogram
    hist, xedges, yedges = np.histogram2d(deg_info, nbr_info, bins=[deg_bin_edges, nbr_bin_edges])

    return hist



def fill_bin_dict(values, bin_dict, midpoint):
    key = get_interval(values[midpoint], bin_dict.keys())
    #return_val = values[midpoint]
    values[midpoint] = -1
    for val in values:
        if in_interval(val, key):
            # Increase counter for each neighbor that is within the same group as the analysed pixel
            # Here could be magnitude as additional criterion included
            bin_dict[key] += 1
    return True

# Get the interval of the analysed pixel
def get_interval(val, intervals):
    for interval in intervals:
        if interval[0] <= val < interval[1]:
            return interval
    else:
        raise ValueError("Value not in any interval.")

# Check if neighbors are in the same bin as the analysed pixel
def in_interval(val, interval):
    # This excludes zero values, which can dominate homogeneous gradient images
    # In non homogenous cases it should be <= val <
    if interval[0] < val <= interval[1]:
        return True
    else:
        return False