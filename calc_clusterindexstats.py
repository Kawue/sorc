import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.spatial.distance import squareform
from helper.misc import rescale_datamatrix

# To maximize
def silhouette(dmatrix, memb):
	return silhouette_score(dmatrix, memb,  metric="precomputed")


# To maximize
def calinski_harabasz(images, memb, mask_img=None):
    images = np.array(images)
    shape2d = (images.shape[0], images.shape[1]*images.shape[2])
    images_2d = images.reshape(shape2d)
    if type(mask_img) != type(None):
        images_2d = np.array([img[np.where(mask_img.flatten() == 1)] for img in images_2d])
    return calinski_harabasz_score(images_2d, memb)


def misc_stats(dmatrix, images, memb):
    # Maximum cluster size ratio
    cluster_sizes = np.array([len(np.where(memb==i)[0]) for i in range(1, np.amax(memb)+1)])
    max_clustersize_ratio = np.amax(cluster_sizes) / len(images)
    # 99p 1p min max ratio
    p99 = np.percentile(dmatrix, 99, interpolation="lower")
    p1 = np.percentile(dmatrix, 1, interpolation="higher")
    dmatrix_p99_1 = rescale_datamatrix(dmatrix, new_min=p1 , new_max=p99)
    np.fill_diagonal(dmatrix_p99_1, 0)
    global_min_max_ratio_99_1 = np.min(squareform(dmatrix_p99_1)) / np.max(squareform(dmatrix_p99_1))
    # 95p 5p min max ratio
    p95 = np.percentile(dmatrix, 95, interpolation="lower")
    p5 = np.percentile(dmatrix, 5, interpolation="higher")
    dmatrix_p95_5 = rescale_datamatrix(dmatrix, new_min=p5 , new_max=p95)
    np.fill_diagonal(dmatrix_p95_5, 0)
    global_min_max_ratio_95_5 = np.min(squareform(dmatrix_p95_5)) / np.max(squareform(dmatrix_p95_5))
    return max_clustersize_ratio, global_min_max_ratio_99_1, global_min_max_ratio_95_5
    

def sorc_stats(dmatrix, images, memb, mask_img, measure_name, eval_object):
    cl_object = {}
    chs = calinski_harabasz(images, memb, mask_img)
    shs = silhouette(dmatrix, memb)
    cl_object["calinski_harabasz_score(h)"] = chs
    cl_object["silhouette_score(h)"] = shs

    max_clustersize_ratio, global_min_max_ratio_99_1, global_min_max_ratio_95_5 = misc_stats(dmatrix, images, memb)
    cl_object["max_clustersize_ratio"] = max_clustersize_ratio
    cl_object["global_min_max_ratio_95_5(l)"] = global_min_max_ratio_95_5
    cl_object["global_min_max_ratio_99_1(l)"] = global_min_max_ratio_99_1

    eval_object[measure_name] = cl_object
    
    return eval_object


def produce_dummy_stats(measure_name, eval_object):
    cl_object = {}
    cl_object["calinski_harabasz_score(h)"] = np.nan
    cl_object["davies_bouldin_score(l)"] = np.nan
    cl_object["max_clustersize_ratio"] = np.nan
    cl_object["global_min_max_ratio_99_1(l)"] = np.nan
    cl_object["global_min_max_ratio_95_5(l)"] = np.nan
    eval_object[measure_name] = cl_object
    return eval_object