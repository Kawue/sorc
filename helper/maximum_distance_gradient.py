import numpy as np
import matplotlib.pyplot as plt

# Question: What happens in cases of more than one maximum?

# Paper: Maximum distance-gradient for robust image registration, 2008
# Provides bonus information in smooth regions.
# Instead of normal gradient maps each vector points towards a source within its neighborhood, which states the highest rate of change.
def maximum_distance_gradient(X, nbrh, gradient_maps=False):
    X = X.copy()
    padded_map = np.pad(X, nbrh, "constant", constant_values=0)
    pos_map = position_map(X.shape, nbrh)
    stack_map = np.dstack((padded_map, pos_map))
    mdg_map = np.empty((*X.shape,2))
    # Iterate over indices of the original array within the padded array
    for i in np.arange(X.shape[0])+nbrh:
        for j in np.arange(X.shape[1])+nbrh:
            # Original z layers are between nbrh and -nbrh, every other layer is padding.
            s_cube = stack_map[i-nbrh:i+nbrh+1, j-nbrh:j+nbrh+1]
            # Caclulate intensity difference
            sp = calc_source_points(s_cube)
            mdg_map[i-nbrh,j-nbrh,:] = sp
    return (mdg_map, mdg_map[:,:,0], mdg_map[:,:,1]) if gradient_maps else mdg_map

def calc_source_points(cube):
    mp = (cube.shape[0]//2, cube.shape[1]//2)
    max_val = 0
    for i in np.arange(cube.shape[0]):
        for j in np.arange(cube.shape[1]):
            intensity = cube[mp[0], mp[1], 0] - cube[i, j, 0]
            direction = cube[mp[0], mp[1], [1, 2]] - cube[i, j, [1, 2]]
            magnitude = np.sum(direction**2)
            delta = intensity*(direction/magnitude)
            delta[np.isnan(delta)] = 0
            delta_mag = np.sqrt(np.sum(delta**2))
            if delta_mag > max_val:
                max_delta = delta
                max_val = delta_mag
    if max_val == 0:
        max_delta = (0,0)
    return np.array(max_delta)


# Creates a 2D map with positions vectors, i.e. Dimensionalities are v,h,2, where m and n are the image height and width.
def position_map(shape, nbrh):
    v = shape[0]
    h = shape[1]
    vi = np.arange(shape[0])
    hi = np.arange(shape[1])
    vi_map = np.empty((v,h))
    vi_map[:] = vi[:,None]
    hi_map = np.empty((v,h))
    hi_map[:] = hi
    vi_map = np.pad(vi_map, nbrh, "constant", constant_values=0)
    hi_map = np.pad(hi_map, nbrh, "constant", constant_values=0)
    pos_map = np.dstack((vi_map, hi_map))
    return pos_map


# maximum distance gradient magnitude map
def mdg_magnitude_map(mdg_map):
    dy = mdg_map[:,:,0]
    dx = mdg_map[:,:,1]
    magnitude_map = np.sqrt(dy**2 + dx**2)
    return magnitude_map


# maximum distance gradient direction map
def mdg_direction_map(mdg_map, degrees):
    dy = mdg_map[:,:,0]
    dx = mdg_map[:,:,1]
    direction_map = np.arctan2(dy, dx)
    return np.degrees(direction_map) if degrees else direction_map