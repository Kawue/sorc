import numpy as np
from numpy.linalg import norm

# Reference: "Local structure orientation descriptor based on intra-image similarity for multimodal registration of liver ultrasound and MR images"
# Can be understood as an alternative analogy of gradient images for noisy images
def lsod(img, window_radius=1):
    img = img.copy()
    padsize = window_radius * 2
    img = np.pad(img, pad_width=padsize, mode="constant", constant_values=0)
    
    if window_radius < 2:
        lfm = calc_lfm(img, window_radius)
    else:
        lfm = calc_lfm_quad(img, window_radius)
    
    lfm_norm = norm(lfm,axis=2)    
    lfm_norm[np.isnan(lfm_norm)] = 0
    
    lfm_normed = lfm / lfm_norm[:,:,None]
    lfm_normed[np.isnan(lfm_normed)] = 0

    #subroutine for R >=2
    return lfm_normed, lfm_normed.sum(axis=2)


def calc_lfm(img,s):
    # patch size (s) is half of the pad size (z)
    z = s*2
    img = img.copy()

    # lsod fature matrix
    lfm = np.zeros((img.shape[0] - z, img.shape[1] - z, (s*2+1)**2))
    # This includes d(x,x) in every feature vector, which may be excluded.
    for i in range(z, img.shape[0]-z):
        for j in range(z, img.shape[1]-z):
            # patch prime
            pp = img[i-s:i+s+1, j-s:j+s+1]
            pp_mean = np.mean(pp)
            # pp vector
            pp_v = []
            for k in range(i-s, i+s+1):
                for l in range(j-s, j+s+1):
                    # patch compare
                    pc = img[k-s:k+s+1, l-s:l+s+1]
                    pp_v.append((pp_mean - np.mean(pc))**2)
            else:
                lfm[i,j] = np.array(pp_v)
    return lfm

def calc_lfm_quad(img,s):
    # patch size (s) is half of the pad size (z)
    z = s*2
    img = img.copy()

    # lsod fature matrix
    lfm = np.zeros((img.shape[0] - z, img.shape[1] - z, (s*2+1)**2))
    # This includes d(x,x) in every feature vector, which may be excluded.
    for i in range(z, img.shape[0]-z):
        for j in range(z, img.shape[1]-z):
            # patch prime
            pp = img[i-s:i+s+1, j-s:j+s+1]
            pp_tl_mean = np.mean(pp[ : s    ,    : s    ])
            pp_tr_mean = np.mean(pp[ : s    , s  : s*2+1])
            pp_bl_mean = np.mean(pp[s: s*2+1,    : s+1  ])
            pp_br_mean = np.mean(pp[s: s*2+1, s+1: s*2+1])
            # pp vector
            pp_v = []
            for k in range(i-s, i+s+1):
                for l in range(j-s, j+s+1):
                    # patch compare
                    pc = img[k-s:k+s+1, l-s:l+s+1]
                    pc_tl_mean = np.mean(pc[ : s    ,    : s    ])
                    pc_tr_mean = np.mean(pc[ : s    , s  : s*2+1])
                    pc_bl_mean = np.mean(pc[s: s*2+1,    : s+1  ])
                    pc_br_mean = np.mean(pc[s: s*2+1, s+1: s*2+1])
                    pp_v.append(np.sum([
                        (pp_tl_mean - pc_tl_mean)**2,
                        (pp_tr_mean - pc_tr_mean)**2,
                        (pp_bl_mean - pc_bl_mean)**2,
                        (pp_br_mean - pc_br_mean)**2
                    ]))
            else:
                lfm[i,j] = np.array(pp_v)
    return lfm

# Paper reference differs in this calculation. There a integration is used.
# The higher the value the better. Range: [0,1]
def compare_lsod(lfm1, lfm2):
    lfm1, lfm2 = lfm1.copy(), lfm2.copy()

    # Change from original paper
    idx1 = np.where(lfm1.sum(axis=2) != 0)
    idx2 = np.where(lfm2.sum(axis=2) != 0)
    zipped1 = list(zip(idx1[0],idx1[1]))
    zipped2 = list(zip(idx2[0],idx2[1]))
    zipped = zipped1 + zipped2
    unzipped = list(zip(*zipped))
    
    h, b = lfm1.shape[0], lfm1.shape[1]
    lfm1 = lfm1.reshape(lfm1.shape[0]*lfm1.shape[1], lfm1.shape[2])
    lfm2 = lfm2.reshape(lfm2.shape[0]*lfm2.shape[1], lfm2.shape[2])
    
    dot_map = np.einsum('ij,ij->i', lfm1, lfm2)
    dot_map = dot_map.reshape(h, b)
    # Change from original paper
    score = np.mean(dot_map[unzipped])
    # TODO: Der Score muss nochmal angeguckt werde, identische Bilder sind weder 0 noch 1
    return score, dot_map