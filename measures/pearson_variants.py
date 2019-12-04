import sys
sys.path.append('.')
import numpy as np
from scipy.stats import pearsonr
from helper.gradient_helper import gradient_map, magnitude_map, direction_map
from measures.shared_residual_similarity import shared_residual_similarity

def pearson(X, Y, index_mask):
    X = X[index_mask]
    Y = Y[index_mask]
    return pearsonr(X, Y)[0]

def pearson_mf(X, Y, index_mask):
    Xdy, Xdx = gradient_map(X)
    Ydy, Ydx = gradient_map(Y)
    Xm = magnitude_map(Xdy, Xdx)
    Ym = magnitude_map(Ydy, Ydx)
    Xo = direction_map(Xdy, Xdx)+180
    Yo = direction_map(Ydy, Ydx)+180

    X = X[index_mask]
    Y = Y[index_mask]
    Xm = Xm[index_mask]
    Ym = Ym[index_mask]
    Xo = Xo[index_mask]
    Yo = Yo[index_mask]

    pi = pearsonr(X, Y)[0]
    pm = pearsonr(Xm, Ym)[0]
    po = pearsonr(Xo, Yo)[0]
    score = np.mean([pi,pm,po])
    return score

def pearson_weighted(X, Y, index_mask):
    _, _, _, w,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*w
        Yw = Y*w
        sscore = pearson(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore

def pearson_weightedplus(X, Y, index_mask):
    _, _, _, wz,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*wz
        Yw = Y*wz
        if (wz == 0).all():
            sscore = -1
        else:
            sscore = pearson(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore