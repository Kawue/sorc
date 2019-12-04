import sys
sys.path.append('.')
import numpy as np
from helper.gradient_helper import gradient_map

# Smooth alternative to gradient image
# Reference: Snakes, Shapes, and Gradient Vector Flow
# t: time step for each iteration
# th: cutoff threshold for stagnation between two time steps
def flow_field(img, t, th):
    def calc_iteration(z,b,c,t):
        x = 1
        y = 1
        mu = 0.2
        if t > (x*y)/(4*mu):
            raise ValueError("t is too big, must be smaller or equal than: " + str((x*y)/(4*mu)))
        r = (mu*t)/(x*y)
        if r > 0.25:
            raise ValueError("r is too big, it must be smaller than 0.25. Change any of the parameters mu, t, x, y.")
        z_pad = np.pad(z,1,"constant",constant_values=0)
        z_nbrs = np.roll(z_pad,1,axis=0)[1:-1,1:-1] + np.roll(z_pad,-1,axis=0)[1:-1,1:-1] + np.roll(z_pad,1,axis=1)[1:-1,1:-1] + np.roll(z_pad,-1,axis=1)[1:-1,1:-1]
        return (1-b*t)*z + r*(z_nbrs - 4*z) + c*t

    if isinstance(th, int):
        max_counter = th
        # Default alternative for ending before max counter is reached
        th = 0.01
    else:
        max_counter = np.inf

    dy, dx = gradient_map(img)
    b = dx**2 + dy**2
    cx = b*dx
    cy = b*dy
    u = dx
    v = dy

    udifflist = []
    vdifflist = []
    
    counter = 0
    while True:
        ul = np.abs(u).sum()
        vl = np.abs(v).sum()
        u = calc_iteration(u,b,cx,t)
        v = calc_iteration(v,b,cy,t)
        udiff = np.abs(np.abs(u).sum() - ul)
        vdiff = np.abs(np.abs(v).sum() - vl)

        if counter < max_counter:
            if np.any((
                    udiff == 0,
                    udiff > th,
                    vdiff == 0,
                    vdiff > th)):
                counter += 1
                udifflist.append(np.abs(u).sum() - ul)
                vdifflist.append(np.abs(v).sum() - vl)
            else:
                break
        else:
            break

    # v, u can basically be used as dy, dx
    return v,u