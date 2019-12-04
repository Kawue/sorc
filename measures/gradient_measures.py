import sys
sys.path.append('.')
import numpy as np
from helper.gradient_helper import gradient_map, magnitude_map, dot_product_map
from helper.histogram_helper import histogram1d, histogram2d
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import resample


# Basis for GI, GC and GO: 3D–2D image registration for target localization in spine surgery: investigation of similarity metrics providing robustness to content mismatch
def gradient_information(img1, img2, index_mask):
    dy1, dx1 = gradient_map(img1)
    gmag1 = magnitude_map(dy1, dx1)
    dy2, dx2 = gradient_map(img2)
    gmag2 = magnitude_map(dy2, dx2)

    vectormap1 = np.dstack((dx1,dy1))
    vectormap2 = np.dstack((dx2,dy2))

    hr_map1 = (vectormap1[:,:,0] == 0) * (vectormap1[:,:,1] == 0)
    hr_map2 = (vectormap2[:,:,0] == 0) * (vectormap2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)

    cos = dot_product_map(vectormap1, vectormap2)
    cos[idx_hr12] = 1
    weight = (cos + 1) / 2
    min_mag = np.amin(np.dstack((gmag1, gmag2)), axis=2)

    # Avoid division by 0
    if min_mag[index_mask].sum() == 0:
        return 0
    else:
        return (weight * min_mag)[index_mask].sum() / min_mag[index_mask].sum()


def gradient_correlation(img1, img2, index_mask):
    dy1, dx1 = gradient_map(img1)
    dy2, dx2 = gradient_map(img2)
    
    vectormapdx1 = np.dstack((dx1,img1))
    vectormapdx2 = np.dstack((dx2,img2))
    vectormapdy1 = np.dstack((img1,dy1))
    vectormapdy2 = np.dstack((img2,dy2))

    hr_map1 = (vectormapdx1[:,:,0] == 0) * (vectormapdx1[:,:,1] == 0)
    hr_map2 = (vectormapdx2[:,:,0] == 0) * (vectormapdx2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)
    cosdx = dot_product_map(vectormapdx1, vectormapdx2)
    cosdx[idx_hr12] = 1

    hr_map1 = (vectormapdy1[:,:,0] == 0) * (vectormapdy1[:,:,1] == 0)
    hr_map2 = (vectormapdy2[:,:,0] == 0) * (vectormapdy2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)
    cosdy = dot_product_map(vectormapdy1, vectormapdy2)
    cosdy[idx_hr12] = 1

    # Adjusted from original source to scale into [0,1]
    return (0.5*(cosdx[index_mask].sum() + cosdy[index_mask].sum())) / cosdx[index_mask].size


# Nlb: Lower bound of evaluated pixel to prevent degenerated solutions.
# t1: threshold to exclude small magnitudes for image one, defaults to 10th percentile.
# t2: threshold to exclude small magnitudes for image two, defaults to 10th percentile.
def gradient_orientation(img1, img2, index_mask, Nlb=10, threshold=True, t1=None, t2=None):
    dy1, dx1 = gradient_map(img1) 
    gmag1 = magnitude_map(dy1, dx1)
    
    dy2, dx2 = gradient_map(img2)
    gmag2 = magnitude_map(dy2, dx2)
    
    if threshold:
        if not t1:
            t1 = np.percentile(gmag1, 10)
        if not t2:
            t2 = np.percentile(gmag2, 10)
            
        # Magnitude maps are needed for threshold based pixel selection.
        gmag1 = [gmag1<t1]
        gmag2 = [gmag2<t2]
        tmap = gmag1 * gmag2

    vectormap1 = np.dstack((dx1,dy1))
    vectormap2 = np.dstack((dx2,dy2))

    hr_map1 = (vectormap1[:,:,0] == 0) * (vectormap1[:,:,1] == 0)
    hr_map2 = (vectormap2[:,:,0] == 0) * (vectormap2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)
    
    cos = dot_product_map(vectormap1, vectormap2)
    cos[idx_hr12] = 1
    logterm = np.log(np.abs(np.arccos(cos)) + 1)
    sumterm = (2 - logterm) / 2

    if threshold:
        tmap_index = np.where(tmap)
        mask_tmap_index = (np.intersect1d(index_mask[0], tmap_index[0]), np.intersect1d(index_mask[1], tmap_index[1]))
        return sumterm[mask_tmap_index].sum() / max(sumterm[mask_tmap_index].size, Nlb) # sumterm might over the whole mask_index, not clear from reference.
    else:
        return sumterm[index_mask].sum() / max(sumterm[index_mask].size, Nlb)


# Position, Magnitude and Angular measure combined
# Basis Paper: Similarity Measure for Vector Field Learning
# Own adaption: Including pixel intensity
def int_mag_an(img1, img2, index_mask, a=(1/3), b=(1/3), pxint=True, c=(1/3), mark_homogeneous=1):
    if pxint:
        if a + b + c != 1:
                raise ValueError("a, b and c must sum to one!")
    else:
        if c and c > 0:
            raise ValueError("If pxint is False factor c will be ignored!")
        if a + b != 1:
            raise ValueError("a and b must sum to one!")
    
    dy1, dx1 = gradient_map(img1) 
    gmag1 = magnitude_map(dy1, dx1)
    
    dy2, dx2 = gradient_map(img2)
    gmag2 = magnitude_map(dy2, dx2)
    
    # Angular measure
    vectormap1 = np.dstack((dx1,dy1))
    vectormap2 = np.dstack((dx2,dy2))
    ang = dot_product_map(vectormap1, vectormap2, mark_homogeneous=mark_homogeneous)
    hr_map1 = (vectormap1[:,:,0] == 0) * (vectormap1[:,:,1] == 0)
    hr_map2 = (vectormap2[:,:,0] == 0) * (vectormap2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)
    ang[idx_hr12] = 1

    # Constant to avoid division by 0
    const = 0.00001

    # Magnitude measure
    mag = (2*gmag1*gmag2 + const) / (gmag1**2 + gmag2**2 + const)

    # Not sclaing ang and use multiplication like ang * (a*mag + c*pxmag) would have the interesting effect that a negative ang makes the whole term negative.
    if pxint:
        pxmag = (2*img1*img2 + const) / (img1**2 + img2**2 + const)
        #return a*np.e**(1-((ang+1)/2)) + b*np.e**(-mag) + c*np.e**(-pxmag)
        # Scale dot product in [0,1] to have the same value ranges for all measures.
        sim_map = a*((ang+1)/2) + b*mag + c*pxmag
    else:
        #return a*np.e**(1-((ang+1)/2)) + b*np.e**(-mag)
        sim_map = a*((ang+1)/2) + b*mag
    score = np.mean(sim_map[index_mask])
    return score, sim_map


def grad_dot_prod(X, Y):
    Xdy, Xdx = gradient_map(X)
    Ydy, Ydx = gradient_map(Y)
    Xv = np.dstack((Xdx, Xdy))
    Yv = np.dstack((Ydx, Ydy))

    # Areas without gradient information are ingored.
    # One could also consider to rate those areas as zero.
    dpm = dot_product_map(Xv, Yv, mark_homogeneous=np.nan)
    index_mask = np.where(np.isfinite(dpm))

    Xm = magnitude_map(Xdy, Xdx)
    Ym = magnitude_map(Ydy, Ydx)
    m = np.abs(Xm - Ym)

    sim_map = (dpm*(1-m))

    # This happens if there are no overlapping gradient regions.
    # One have to decide if this equals 0 or -1.
    # Here we decided for 0 as this stands for no relation at all.
    if np.array(index_mask).size > 0:
        sim_pooled = np.mean(sim_map[index_mask])
    else:
        sim_pooled = 0
    return sim_pooled
    

# Both images get their own 2dHistogram, which can then be compared with histogram methods.
def grad_hist_dot(img1, img2, sample_number=None):
    sample_hardcap = 10000
    rs = 0 # For reproduction purposes, otherwise choose None

    dy1, dx1 = gradient_map(img1)
    dy2, dx2 = gradient_map(img2)
    
    vectormap1 = np.dstack((dx1,dy1)).reshape(img1.shape[0] * img1.shape[1], 2)
    vectormap2 = np.dstack((dx2,dy2)).reshape(img2.shape[0] * img2.shape[1], 2)

    if sample_number:
        vectormap1 = resample(vectormap1, replace=False, n_samples=sample_number, random_state=rs)
        vectormap2 = resample(vectormap2, replace=False, n_samples=sample_number, random_state=rs)
    else:
        sample_number = sample_hardcap
        if vectormap1.shape[0] > sample_hardcap:
            print("For computational and memory reasons we have a hardcap of " + str(sample_hardcap) + " randomly sampled pixel. Image one will be resampled.")
            vectormap1 = resample(vectormap1, replace=False, n_samples=sample_number, random_state=rs)
        if vectormap2.shape[0] > sample_hardcap:
            print("For computational and memory reasons we have a hardcap of " + str(sample_hardcap) + " randomly sampled pixel. Image two will be resampled.")
            vectormap2 = resample(vectormap2, replace=False, n_samples=sample_number, random_state=rs)

    inner1 = np.inner(vectormap1, vectormap1)
    inner2 = np.inner(vectormap2, vectormap2)

    euclidean1 = euclidean_distances(vectormap1, vectormap1)
    euclidean2 = euclidean_distances(vectormap2, vectormap2)

    idx1 = np.where(inner1 != 0)
    idx2 = np.where(inner2 != 0)
    inner1 = inner1[idx1]
    inner2 = inner2[idx2]
    euclidean1 = euclidean1[idx1]
    euclidean2 = euclidean2[idx2]

    hist_inner1, bins_inner1 = histogram1d(inner1)
    hist_inner2, bins_inner2 = histogram1d(inner2)
    hist_euclidean1, bins_euclidean1 = histogram1d(euclidean1)
    hist_euclidean2, bins_euclidean2 = histogram1d(euclidean2)

    hist2d1, xbins_1, ybins_1 = histogram2d(euclidean1, inner1, bins_euclidean1, bins_inner1) 
    hist2d2, xbins_2, ybins_2 = histogram2d(euclidean2, inner2, bins_euclidean2, bins_inner2)

    return hist2d1, hist2d2, xbins_1, ybins_1, xbins_2, ybins_2


# Both images get their own 2dHistogram, which can then be compared with histogram methods.
def grad_hist_curv(img1, img2, dy1=None, dy2=None, dx1=None, dx2=None, sample_number=None):
    sample_hardcap = 10000
    rs = 0 # For reproduction purposes, otherwise choose None
    
    dy1, dx1 = gradient_map(img1)
    dy2, dx2 = gradient_map(img2)

    # Row-wise pixel reshape into a 1D-vector, i.e. (x0y0, ..., xny0, x0y1, ..., xny1, ..., x0yn, ..., xnyn)
    vectormap1 = np.dstack((dx1,dy1)).reshape(img1.shape[0] * img1.shape[1], 2)
    vectormap2 = np.dstack((dx2,dy2)).reshape(img2.shape[0] * img2.shape[1], 2)

    dyx1, dxx1 = gradient_map(dx1)
    dyy1, dxy1 = gradient_map(dy1)
    dyx2, dxx2 = gradient_map(dx2)
    dyy2, dxy2 = gradient_map(dy2)

    # Calculate curvature
    curv1 = np.abs(dx1*dyy1 - dy1*dxx1) / (dx1**2 + dy1**2)**(3/2)
    curv1[np.isnan(curv1)] = 0
    curv1 = curv1.flatten()
    curv2 = np.abs(dx2*dyy2 - dy2*dxx2) / (dx2**2 + dy2**2)**(3/2)
    curv2[np.isnan(curv2)] = 0
    curv2 = curv2.flatten()

    if sample_number:
        vectormap1 = resample(vectormap1, replace=False, n_samples=sample_number, random_state=rs)
        vectormap2 = resample(vectormap2, replace=False, n_samples=sample_number, random_state=rs)

        curv1 = resample(curv1, replace=False, n_samples=sample_number, random_state=rs)
        curv2 = resample(curv2, replace=False, n_samples=sample_number, random_state=rs)
    else:
        sample_number = sample_hardcap
        if vectormap1.shape[0] > sample_hardcap:
            print("For computational and memory reasons we have a hardcap of " + str(sample_hardcap) + " randomly sampled pixel. Image one will be resampled.")
            vectormap1 = resample(vectormap1, replace=False, n_samples=sample_number, random_state=rs)
            curv1 = resample(curv1, replace=False, n_samples=sample_number, random_state=rs)
        if vectormap2.shape[0] > sample_hardcap:
            print("For computational and memory reasons we have a hardcap of " + str(sample_hardcap) + " randomly sampled pixel. Image two will be resampled.")
            vectormap2 = resample(vectormap2, replace=False, n_samples=sample_number, random_state=rs)
            curv2 = resample(curv2, replace=False, n_samples=sample_number, random_state=rs)

    # Calculate pairwise mean curvature (same shape as vectormap)
    curv1 = (curv1[:,None] + curv1) / 2
    curv2 = (curv2[:,None] + curv2) / 2

    euclidean1 = euclidean_distances(vectormap1, vectormap1)
    euclidean2 = euclidean_distances(vectormap2, vectormap2)

    hist_curv1, bins_curv1 = histogram1d(curv1)
    hist_curv2, bins_curv2 = histogram1d(curv2)
    hist_euclidean1, bins_euclidean1 = histogram1d(euclidean1)
    hist_euclidean2, bins_euclidean2 = histogram1d(euclidean2)

    hist2d1, xbins_1, ybins_1 = histogram2d(euclidean1, curv1, bins_euclidean1, bins_curv1)
    hist2d2, xbins_2, ybins_2 = histogram2d(euclidean2, curv2, bins_euclidean2, bins_curv2)

    return hist2d1, hist2d2, xbins_1, ybins_1, xbins_2, ybins_2