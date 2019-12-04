import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sea
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def quiver_plot(dx, dy, scale):
    dx, dy = dx.copy(), dy.copy()

    #plt.figure()
    ax = plt.gca()
    # Plot the gradient fielda
    plt.quiver(dx, dy, angles="xy", scale=scale)
    # Set the origin of the gradient field equal to the origin of the image
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    ax.set_aspect("equal")


# TODO: Remove exclude zero later when masking is available
def histogram1d_plot(hist1d, bins, exclude_zero=False):
    hist1d = hist1d.copy()

    if exclude_zero:
        hist1d[0] = 0
    #plt.figure()
    bin_center = (bins[:-1] + bins[1:])/2
    bin_width = (bins[1] - bins[0]) * 0.8
    plt.bar(bin_center, height=hist1d, width=bin_width, align="center")
    #plt.show()


# TODO: Remove exclude zero later when masking is available
def histogram2d_plot(hist2d, binsX, binsY, exclude_zero=False):
    hist2d = hist2d.copy()

    # Reason for flip: hist2d index begins from left to right and top to bottom. For visual purposes we would like to show it from left to right and top to bottom.
    if exclude_zero:
        hist2d[0,0] = 0

    hist2d = np.flip(hist2d, axis=0)
    aximg = plt.subplot(111)
    img = aximg.imshow(hist2d, extent=[binsY[0], binsY[-1], binsX[0], binsX[-1]])
    aximg.set_aspect(1)

    divider = make_axes_locatable(aximg)
    axHistX = divider.append_axes("right", size=1.2, pad=0.6, sharey=aximg)
    plt.title("X")
    axHistY = divider.append_axes("top", size=1.2, pad=0.6, sharex=aximg)
    plt.title("Y")
    axColorbar = divider.append_axes("left", size=0.25, pad=0.8)
    
    plt.colorbar(img, cax=axColorbar)
    axColorbar.yaxis.tick_left()
    
    # xhist_sum needs to be inverted for the following reason:
    # Plotting with barh fills the blot from bottom to top.
    # After flipping above te first value in xhist_sum corresponds to the last index of the unflipped hist2d.
    # After inverting the first value of xhist_sum, i.e. the lowest bar in barh corresponds to the first index of the unflipped hist2d, i.e. the lowest row in the flipped imshow.
    xhist_sum = hist2d.sum(axis=1)[::-1]
    bin_height = binsX[1] - binsX[0]
    bin_centers = (binsX[:-1] + binsX[1:]) / 2
    axHistX.barh(bin_centers, width=xhist_sum, height=bin_height, align="center")
    
    
    yhist_sum = hist2d.sum(axis=0)
    bin_width = binsY[1] - binsY[0]
    bin_centers = (binsY[:-1] + binsY[1:]) / 2
    axHistY.bar(bin_centers, height=yhist_sum, width=bin_width, align="center")
    #for label in axHistY.get_xticklabels():
    #    label.set_rotation("vertical")
    
    #aximg.get_xaxis().set_visible(False)
    #aximg.get_yaxis().set_visible(False)
    #plt.show()


# Compensational Method which calculated the joint Histogram by itself until my own method above is fixed.
def seaborn_histogram2d_plot(X, Y):
    X, Y = X.copy(), Y.copy()

    df = pd.DataFrame(np.array([X.flatten(),Y.flatten()]), columns=["X","Y"])
    sea.jointplot("X", "Y", data=df, kind="hex", stat_func=None)