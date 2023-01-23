import sys
sys.path.append('.')
from sys import argv
import argparse
import os
import json
import warnings
import pandas as pd
import numpy as np
from time import time
from helper.read import read_images, index_mask
from helper.scalespace import preprocess_scalespace
from helper.preprocessing import preprocess_images
from helper.image_write_helper import save_clustered_images, intra_cluster_boxplot, cluster_evaluation
from helper.misc import str2bool
from calc_sd_matrix import create_distance_matrix, similarity_distance_switcher
from calc_cluster import Hierarchical, AffinityPropagation, MsiCommunityDetectionPCA, MsiCommunityDetectionStatistics, kMedoidsEM, kMedoidsPAM
from calc_clusterindexstats import sorc_stats, produce_dummy_stats
from demo_calls import calc_pearson_ms, calc_cosine_ms, calc_angular_ms, calc_ssim_ms, calc_gssim_ms, calc_multi_feature_similarity, calc_shared_residual_similarity, calc_hypergeometric_similarity, calc_contingency_similarity, calc_mdsi_similarity, calc_local_std_similarity, calc_int_mag_an, calc_gradient_information, calc_intensity_hist, calc_MI, calc_fsim_ms, calc_fsm_ms, calc_gmsd_ms, calc_stsim_ms, calc_ssim4_ms, calc_gssim4_ms
from sorc import write_dframe, write_sorc

# For new Measures, Change json, json_mapper, finalmethods, title_prettify and soup

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--imagespath", type=str, required=True, help="Path to images folder.")
parser.add_argument("-d", "--h5path", type=str, required=True, help="Path to h5 file.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
parser.add_argument("-n", "--dataname", type=str, required=False, help="Name for the main save folder.")
parser.add_argument("-p", "--padding", type=int, required=False, default=13, help="Padding of images. Also defines the window size for window based measures.")
parser.add_argument("-c", "--clusternumber", type=int, required=False, help="Number of clusters. Only needed if the clustering procedure requires a pre defined number.")
#parser.add_argument("-cd", "--communitydetection", type=str, required=False, choices=["pca", "statistics"], help="Edge reduction method for msi community detection method.")
parser.add_argument("-pp", "--applypreprocess", required=False, action='store_true', default=False, help="Apply preprocessing?.")
parser.add_argument("-ss", "--applyscalespace", required=False, action='store_true', default=False, help="Apply scale space?.")
parser.add_argument("-sz", "--stepsize", type=int, required="-ss" in argv or "--applyscalespace" in argv, choices=[1,2], help="Step size for scale space images.")
parser.add_argument("-pl", "--plotall", required=False, action='store_true', default=False, help="Plot all images with directories organized according to the cluster structure.")
args=parser.parse_args()

pp_ms_parameters = ""
if args.applypreprocess:
    pp_ms_parameters += "PP"
else:
    pp_ms_parameters += "noPP"
if args.applyscalespace:
    pp_ms_parameters += str(args.stepsize) + "MS"
else:
    pp_ms_parameters += "noMS"

# Set argument variables
dirpath = args.imagespath
h5path = args.h5path
padding = args.padding
win_size = args.padding
savepath = args.savepath
apply_preprocess = args.applypreprocess
apply_scalespace = args.applyscalespace
stepsize = args.stepsize
nr_cluster = args.clusternumber
#er_method = args.communitydetection
plotall = args.plotall

if args.dataname:
    dataname = args.dataname
else:
    dataname = os.path.basename(args.h5path).split(".")[0]

# Read all necessary data
dframe = pd.read_hdf(h5path)
imgs, img_names = read_images(dirpath, padding)
index_mask, mask_img = index_mask(dframe, padding)
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "measures.json"), "r") as read_file:
    setup_json = json.load(read_file)

# Preprocess data
if apply_preprocess:
    imgs = preprocess_images(imgs, index_mask, mask_img)
    
if apply_scalespace:
    if stepsize == 1:
        imgs_ms = preprocess_scalespace(imgs, "gauss", 3)
    elif stepsize == 2:
        imgs_ms = preprocess_scalespace(imgs, "gauss", 5)[[0,2,4]]
    else:
        raise ValueError("Workflow is predefined for 0, 1 or 2 step size scale space.")
else:
    if stepsize:
        raise ValueError("Scale space must be applied to use the step size parameter.")

json_mapper = {
    "Pearson MS": [calc_pearson_ms, True, {}],
    "Cosine MS": [calc_cosine_ms, True, {}],
    "Angular MS": [calc_angular_ms, True, {}],
    "SSIM MS": [calc_ssim_ms, True, {"win_size": padding, "sigma": None}],
    "GSSIM MS": [calc_gssim_ms, True, {"win_size": padding, "sigma": None}],
    "GMSD MS": [calc_gmsd_ms, True, {}],
    "STSIM MS": [calc_stsim_ms, True, {"win_size": padding, "sigma": None}],
    "MFS Max MS": [calc_multi_feature_similarity, True, {"weighted": False, "wplus": False, "pooling": "max", "win_size": padding, "sigma": None}],
    "Shared Pixel": [calc_shared_residual_similarity, False, {"count_zeros": False, "index_mask":index_mask}],
    "Hypergeometric": [calc_hypergeometric_similarity, False, {}],
    "Contingency": [calc_contingency_similarity, False, {}],
    "MDSI Sim": [calc_mdsi_similarity, False, {}],
    "Local Std": [calc_local_std_similarity, False, {"win_size": padding}],
    "IntMagAn": [calc_int_mag_an, False, {}],
    "Grad Info": [calc_gradient_information, False, {}],
    "Mutual Info": [calc_MI, False, {}],
    "Histogram": [calc_intensity_hist, False, {}]
}

stats_dict = {}
for cluster_method in setup_json[0]["ClusterMethods"]:
    stats_dict[cluster_method] = {}

for measure_id in setup_json[0]["Measures"]:
    #timestamppath = os.path.join(savepath, dataname, pp_ms_parameters, measure_id)
    #if not os.path.exists(timestamppath):
    #    os.makedirs(timestamppath)
    #f = open(os.path.join(timestamppath, "computationTimes.txt"), "w+")
    measure = json_mapper[measure_id][0]
    multiscale_save = json_mapper[measure_id][1]
    kwargs = json_mapper[measure_id][2]
    # Template for time measurements
    template = "{:<18} {:<23} {:<50}"
    start = time()
    print()
    print("Start %s Matrix Calculation ..... "%(measure_id), end="")
    if apply_scalespace:
        if multiscale_save:
            dmatrix = create_distance_matrix(imgs_ms, measure, index_mask, kwargs)
        else:
            dmatrix = create_distance_matrix(imgs, measure, index_mask, kwargs)
    else:
        dmatrix = create_distance_matrix(imgs, measure, index_mask, kwargs)
    print("Finished!")
    print()
    end = time()
    #print(template.format("Time for Method |", measure_id, "| " + str(np.around(end-start, 6))), file=f)
    for cluster_method in setup_json[0]["ClusterMethods"]:
        print("Start Clustering with %s ..... "%(cluster_method), end="")
        try:
            if cluster_method == "HierarchicalClustering": #Distance based
                if (np.diag(dmatrix) != 0).all():
                    dmatrix = similarity_distance_switcher(dmatrix)
                cluster = Hierarchical(dmatrix, nr_cluster)
                labels = cluster.perform()
            elif cluster_method == "AffinityPropagationClustering": #Similarity based
                if (np.diag(dmatrix) != 1).all():                    
                    dmatrix = similarity_distance_switcher(dmatrix)
                cluster = AffinityPropagation(dmatrix)
                labels = cluster.perform()
            elif cluster_method == "MsiCommunityDetectionPCA": #Similarity based
                if (np.diag(dmatrix) != 1).all():
                    dmatrix = similarity_distance_switcher(dmatrix)
                cluster = MsiCommunityDetectionPCA(dmatrix, os.path.join(savepath, dataname, pp_ms_parameters, cluster_method, measure_id))
                labels = cluster.perform()
            elif cluster_method == "MsiCommunityDetectionStatistics": #Similarity based
                if (np.diag(dmatrix) != 1).all():
                    dmatrix = similarity_distance_switcher(dmatrix)
                cluster = MsiCommunityDetectionStatistics(dmatrix)
                labels = cluster.perform()
            elif cluster_method == "kMeansEM": #Distance based
                if (np.diag(dmatrix) != 0).all():
                    dmatrix = similarity_distance_switcher(dmatrix)
                dmatrix = similarity_distance_switcher(dmatrix)
                cluster = kMedoidsEM(dmatrix, nr_cluster)
                labels = cluster.perform()
            elif cluster_method == "kMeansPAM": #Distance based
                if (np.diag(dmatrix) != 0).all():
                    dmatrix = similarity_distance_switcher(dmatrix)
                dmatrix = similarity_distance_switcher(dmatrix)
                cluster = kMedoidsPAM(dmatrix, nr_cluster)
                labels = cluster.perform()
            subsavepath = os.path.join(savepath, dataname, pp_ms_parameters, cluster_method, measure_id)
            print("Finished!")
            print()
            print("Start Write Cluster Images with %s ..... "%(cluster_method), end="")
            if plotall:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    save_clustered_images(imgs, img_names, labels, subsavepath)
                    c_arrays = cluster.prep_cluster_similarity_array()
                    intra_cluster_boxplot(c_arrays, labels, subsavepath, measure_id)
                    cluster_evaluation(c_arrays, labels, imgs, subsavepath, measure_id)
            print("Finished!")
            print()
            print("Start SORC Stats Calculation ..... ", end="")
            # Distance based
            if (np.diag(dmatrix) != 0).all():
                dmatrix = similarity_distance_switcher(dmatrix)
            stats_dict[cluster_method] = sorc_stats(dmatrix, imgs, labels, mask_img, measure_id, stats_dict[cluster_method])
            print("Finished!")
        except Exception as e:
            print()
            print("Exception in cluster evaluation. Dummy stats are used instead.")
            print("Exception Message: " + str(e))
            stats_dict[cluster_method] = produce_dummy_stats(measure_id, stats_dict[cluster_method])
        #dmatrix = similarity_distance_switcher(dmatrix)
    #f.close()
    print(stats_dict.keys())

stats_sort_dict = {
    1: ["silhouette_score(h)", 1],
    2: ["calinski_harabasz_score(h)", 1],
    3: ["max_clustersize_ratio", -1],
    4: ["global_min_max_ratio_99_1(l)", -1],
    5: ["global_min_max_ratio_95_5(l)", -1]
}

print("Start SORC Evaluation ..... ", end="")
subsavepath = os.path.join(savepath, dataname)
write_dframe(subsavepath, pp_ms_parameters, stats_dict, stats_sort_dict)
print("Finished SORC Evaluation.")
print()
print("DONE!")

'''
h5pypath = os.path.join(subsavepath, os.path.basename(subsavepath) +  "_sorc_evaluation.h5")
write_sorc(h5pypath, subsavepath)
'''