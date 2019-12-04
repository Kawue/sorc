import sys
sys.path.append('.')
import numpy as np
from helper.entropy_helper import prepare_image_histograms, image_histogram, histogram1d
from measures.local_standard_deviation_based_image_quality import lsdbiq
from measures.entropy_variants import local_entropy_map, hellinger_distance, jensen_shannon_divergence, entropy, mutual_information, conditional_entropy
from measures.hog_variant import hog
from measures.edge_orientation_autocorrelogram import eoac
from measures.mean_deviation_similarity_index import mdsi
from measures.contingency_similarity import contingency
from measures.cosine_variants import cosine
from measures.pearson_variants import pearson
from measures.hypergeometric_similarity import hypergeometric_similarity
from measures.gradient_measures import gradient_information, int_mag_an
from measures.multi_feature_similarity_variants import multi_feature_similarity
from measures.ssim_variants import compare_ssim
from measures.shared_residual_similarity import shared_residual_similarity



def calc_pearson_ms(X, Y, index_mask, kwargs):
    score = pearson(X, Y, index_mask)
    return score
    
def calc_cosine_ms(X, Y, index_mask, kwargs):
    score = cosine(X, Y, index_mask)
    return score

def calc_ssim_ms(X, Y, index_mask, kwargs):
    score, _ = compare_ssim(X, Y, index_mask, **kwargs)
    return score

def calc_multi_feature_similarity(X, Y, index_mask, kwargs):
    score, _ = multi_feature_similarity(X, Y, index_mask, **kwargs)
    return score

def calc_shared_residual_similarity(X, Y, index_mask, kwargs):
    score, _, _, _, _, _, _ = shared_residual_similarity(X, Y, **kwargs)
    return score

def calc_hypergeometric_similarity(X, Y, index_mask, kwargs):
    score = hypergeometric_similarity(X, Y, index_mask)
    return score

def calc_contingency_similarity(X, Y, index_mask, kwargs):
    score = contingency(X, Y, index_mask)
    return score

def calc_mdsi_similarity(X, Y, index_mask, kwargs):
    dev_sim, mdsi_score, mdsi_map = mdsi(X, Y, index_mask=index_mask, emphase=None, return_map=True)
    return 1-dev_sim

def calc_local_std_similarity(X, Y, index_mask, kwargs):
    _, lsdbiq_map = lsdbiq(X, Y, index_mask=index_mask, **kwargs)
    lsdbiq_score = np.mean(lsdbiq_map[index_mask])
    return lsdbiq_score

def calc_int_mag_an(X, Y, index_mask, kwargs):
    score, _ = int_mag_an(X, Y, index_mask)
    return score

def calc_gradient_information(X, Y, index_mask, kwargs):
    score = gradient_information(X, Y, index_mask)
    return score

def calc_intensity_hist(X, Y, index_mask, kwargs):
    Xhist, Xbin = image_histogram(X[index_mask])
    Yhist, Ybin = image_histogram(Y[index_mask])
    int_score = 1-hellinger_distance(Xhist, Yhist)
    return int_score

def calc_MI(X, Y, index_mask, kwargs):
    Hx, Hxbins, Hy, Hybins, Hxy, Hxxybins, Hyxybins = prepare_image_histograms(X[index_mask], Y[index_mask], bins=True)
    # https://stats.stackexchange.com/questions/21317/can-mutual-information-gain-value-be-greater-than-1
    eX = entropy(Hx)
    eY = entropy(Hy)
    if np.amin([eX, eY]) == 0:
        mi = 0
    else:
        mi = mutual_information(Hx, Hy, Hxy) / np.amin([eX, eY])
    return mi































##### Other Example Calls #####

def jsd(X, Y, index_mask):
    Hx, Xbin = image_histogram(X[index_mask])
    Hy, Ybin = image_histogram(Y[index_mask])
    jsdiv = 1-jensen_shannon_divergence(Hx, Hy)
    return jsdiv


def reciprocal_conditional_entropy(X, Y, index_mask):
    Hx, Hxbins, Hy, Hybins, Hxy, Hxxybins, Hyxybins = prepare_image_histograms(X[index_mask], Y[index_mask], bins=True)
    eX = entropy(Hx)
    eY = entropy(Hy)
    # Sums the uncertainty about X if Y is known and about Y if X is known.
    # Normalization results from: https://math.stackexchange.com/questions/496584/maximum-entropy-joint-distribution-from-marginals
    # and: https://en.wikipedia.org/wiki/Conditional_entropy#Other_properties
    # Both conds are 1 if X and Y are independent, resulting in: 1-2 = -1
    if eX == 0:
        eX = 1
    if eY == 0:
        eY = 1
    score = 1 - ( (conditional_entropy(Hx, Hy, Hxy) / eX) + (conditional_entropy(Hy, Hx, Hxy) / eY) )
    return score


def local_entropy(X, Y, wsize, index_mask):
    c = 0.00001
    try:
        X_entropy_map_c = local_entropy_map(X, size=wsize, entropy_type="collision")
        Y_entropy_map_c = local_entropy_map(Y, size=wsize, entropy_type="collision")
    except Exception as e:
        print(e)
        sys.stdout.flush()
    sim_map = (2*X_entropy_map_c*Y_entropy_map_c+c) / (X_entropy_map_c**2 + Y_entropy_map_c**2+c)
    sim_score = np.mean(sim_map[index_mask])
    return sim_score


def local_entropy_hist(X, Y, wsize, index_mask):
    try:
        X_entropy_map_c = local_entropy_map(X, size=wsize, entropy_type="collision")
        Y_entropy_map_c = local_entropy_map(Y, size=wsize, entropy_type="collision")
    except Exception as e:
        print(e)
        sys.stdout.flush()
    emax = np.amax([X_entropy_map_c[index_mask], Y_entropy_map_c[index_mask]])
    emin = np.amin([X_entropy_map_c[index_mask], Y_entropy_map_c[index_mask]])
    X_entropy_hist, X_entropy_bins = histogram1d(X_entropy_map_c[index_mask], bins=int(np.log2(X_entropy_map_c[index_mask].size) + 1), range=(int(emin)-1, int(emax)+1))
    Y_entropy_hist, Y_entropy_bins = histogram1d(Y_entropy_map_c[index_mask], bins=int(np.log2(X_entropy_map_c[index_mask].size) + 1), range=(int(emin)-1, int(emax)+1))
    entropy_score = 1-hellinger_distance(X_entropy_hist, Y_entropy_hist)
    return entropy_score


def hog_sim(X, Y):
    X_hog_descr, X_hog_map = hog(X)
    Y_hog_descr, Y_hog_map = hog(Y)
    hog_descr_score = 1-hellinger_distance(X_hog_descr, Y_hog_descr)
    return hog_descr_score


def eoac_sim(X, Y):
    X_eoac = eoac(X, None)
    Y_eoac = eoac(Y, None)
    eoac_score = 1-hellinger_distance(X_eoac, Y_eoac)
    return eoac_score