import sys
sys.path.append('.')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helper.gradient_helper import gradient_map, magnitude_map, direction_map
from measures.shared_residual_similarity import shared_residual_similarity

def cosine(X, Y, index_mask):
    X = X[index_mask]
    Y = Y[index_mask]
    return cosine_similarity(X[None,:], Y[None,:])[0][0]


def cosine_mf(X, Y, index_mask):
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

    pi = cosine_similarity(X[None,:], Y[None,:])[0][0]
    pm = cosine_similarity(Xm[None,:], Ym[None,:])[0][0]
    po = cosine_similarity(Xo[None,:], Yo[None,:])[0][0]
    score = np.mean([pi, pm,po])
    return score

def cosine_weighted(X, Y, index_mask):
    _, _, _, w,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*w
        Yw = Y*w
        sscore = cosine(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore

def cosine_weightedplus(X, Y, index_mask):
    _, _, _, wz,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*wz
        Yw = Y*wz
        if (wz == 0).all():
            sscore = 0
        else:
            sscore = cosine(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore





def angular(X, Y, index_mask, only_positive=True):
    if only_positive:
        factor = 2
    else:
        factor = 1
    X = X[index_mask]
    Y = Y[index_mask]
    cos = np.around(cosine_similarity(X[None,:], Y[None,:])[0][0], 10)
    return 1 - ((factor * np.arccos(cos)) / np.pi)

def angular_mf(X, Y, index_mask, only_positive=True):
    if only_positive:
        factor = 2
    else:
        factor = 1
        
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

    cosi = np.around(cosine_similarity(X[None,:], Y[None,:])[0][0], 10)
    cosm = np.around(cosine_similarity(Xm[None,:], Ym[None,:])[0][0], 10)
    coso = np.around(cosine_similarity(Xo[None,:], Yo[None,:])[0][0], 10)

    pi = 1 - ((factor * np.arccos(cosi)) / np.pi)
    pm = 1 - ((factor * np.arccos(cosm)) / np.pi)
    po = 1 - ((factor * np.arccos(coso)) / np.pi)
    score = np.mean([pi, pm, po])
    return score

def angular_weighted(X, Y, index_mask):
    _, _, _, w,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*w
        Yw = Y*w
        sscore = angular(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore

def angular_weightedplus(X, Y, index_mask):
    _, _, _, wz,_ , _, _ = shared_residual_similarity(X, Y, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
    try:
        Xw = X*wz
        Yw = Y*wz
        if (wz == 0).all():
            sscore = 0
        else:
            sscore = angular(Xw, Yw, index_mask)
    except Exception as e:
        print(e)
    return sscore