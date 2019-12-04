import numpy as np

def create_distance_matrix(imgs, measure, index_mask, kwargs):
    # Differentiation between singlescale and multiscale.
    # Shape for singlescale [#imgs, dim X, dim Y]
    # Shape for multiscale [#sclaes, #imgs, dim X, dim Y]
    if imgs.ndim == 4:
        dmatrix = np.empty((imgs.shape[1], imgs.shape[1]))
        for i in range(imgs.shape[1]):
            for j in range(imgs.shape[1]):
                Xs = imgs[: ,i]
                Ys = imgs[: ,j]
                try:
                    score = ms_procedure(Xs, Ys, measure, index_mask, kwargs)
                    dmatrix[i][j] = score
                except Exception as e:
                    print("Exception during distance matrix generation.")
                    print(measure)
                    print(e)
                    print()
    elif imgs.ndim == 3:
        dmatrix = np.empty((imgs.shape[0], imgs.shape[0]))
        for i, X in enumerate(imgs):
            for j, Y in enumerate(imgs):
                try:
                    score = measure(X, Y, index_mask, kwargs)
                    dmatrix[i][j] = score
                except Exception as e:
                    print("Exception during distance matrix generation.")
                    print(measure)
                    print(e)
                    print()
    else:
        raise ValueError("Dimension of images array needs to be three (without multiscale) or four (with multiscale) to create a distance matrix.")
    # Convert similarity matrix into distance matrix
    return dmatrix

def similarity_distance_switcher(dmatrix):
    return np.around(1-dmatrix, 7)


def ms_procedure(Xs, Ys, measure, index_mask, kwargs):
    sim_vals = []
    for i in range(Xs.shape[0]):
        X = Xs[i]
        Y = Ys[i]
        score = measure(X, Y, index_mask, kwargs)
        sim_vals.append(score)
    ms_score = np.mean(sim_vals)  
    return ms_score