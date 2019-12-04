import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import squareform
from skimage import img_as_ubyte
import skimage.filters as skfilters
from skimage.morphology import opening, closing, disk
from helper.maximally_stable_regions import calc_mser

# Save clustered images into one folder per cluster
def save_clustered_images(imgs, names, memb, savepath):
    nr_cluster = int(np.amax(memb))
    for i in range(1, nr_cluster+1):
        idx = np.where(memb == i)[0]
        for j in idx:
            if not os.path.exists(os.path.join(savepath, "c" + str(i))):
                os.makedirs(os.path.join(savepath, "c" + str(i)))
                os.makedirs(os.path.join(savepath, "c" + str(i), "additional-images"))
            plt.imsave(os.path.join(savepath, "c" + str(i) , names[j] + ".png"), imgs[j], vmin=0, vmax=1)
            plt.close()



# List of array with similarity values within clusters, cluster labels, method name
def intra_cluster_boxplot(c_arrays, memb, savepath, measure_name):
    fig, ax = plt.subplots()
    plt.boxplot(c_arrays, whis=[5,95], showmeans=True, meanline=True)
    plt.title("Intra-Cluster Distance Values: " + measure_name)
    ax.set_xticklabels(["%s\n$v$=%d\n$n$=%d" % (i, len(v), np.where(memb==i)[0].size) for i,v in enumerate(c_arrays, start=1)])
    plt.xlabel("Clusters")
    plt.ylabel("Distance Values")
    plt.savefig(os.path.join(savepath, "boxplot-" + measure_name))
    plt.close()



def cluster_evaluation(c_arrays, memb, pp_imgs, savepath, measure_name):
    # mean, median, std in similarity values
    # Singleton cluster will appear as NaN, since their c_array is empty.
    mean_dists = [np.mean(x) for x in c_arrays]
    median_dists = [np.median(x) for x in c_arrays]
    std_dists = [np.std(x) for x in c_arrays]
    cluster_sizes = [len(np.where(memb==i)[0]) for i in range(1, np.amax(memb)+1)]
    

    with open(os.path.join(savepath, measure_name) + ".txt", "a") as txt:
        txt.write("Statistical intra cluster analysis. Singletons will appear as NaN.\n")
        for idx, (size, mean, median, std) in enumerate(zip(cluster_sizes, mean_dists, median_dists, std_dists)):
            txt.write("Cluster %d:\nSize: %d, Mean: %f, Median: %f, Std: %f\n" % (idx + 1, size, mean, median, std))
        txt.write("\n\n")
        txt.write("Statistical cluster image stack analysis:\n")
        txt.write("Color scales of mean, median and max aggregation images are bounded in [0,1], while std aggregation images are bounded in [0,0.5], as 0.5 is the maximum standard deviation for values in [0,1].\n")
        txt.write("Hull abstraction is created by Otsu binarizing and opening(closing()) with a disk struct of radius 2.\n")
        txt.write("Region abtraction is created by the maximally stable extremal regions (MSER) algorithm.")
        txt.write("Mean standard deviation (MSD): Standard deviation for each Pixel across the cluster image stack.\n")
        txt.write("Mean absolute deviation (MAD): Mean absolute deviation of every image within the cluster image stack from the clusters mean aggregation image.\n")
        txt.write("MSD and MAD are calculate over every pixel with signal > 0 in at least one image.\n")
        txt.write("\n")

    # mean, median, std difference from representation image
    pp_imgs = np.array(pp_imgs)
    additional_images = "additional-images"
    for grp in range(1, np.amax(memb)+1):
        grp_imgs = pp_imgs[np.where(memb == grp)]
        mean_img = np.mean(grp_imgs, axis=0)
        median_img = np.median(grp_imgs, axis=0)
        std_img = np.std(grp_imgs, axis=0)
        max_img = np.max(grp_imgs, axis=0)
        t_otsu = skfilters.threshold_otsu(max_img)
        hull_img = np.zeros(max_img.shape)
        hull_img[np.where(max_img > t_otsu)] = 1
        sum_img = np.sum(grp_imgs, axis=0)
        sum_img = (sum_img - np.amin(sum_img)) / (np.amax(sum_img) - np.amin(sum_img))
        measured_area_size = int(np.where(sum_img > 0)[0].size * 0.8)
        mser_img, stable_regions = calc_mser(img_as_ubyte(sum_img), 3, 1, 0.01, measured_area_size, True)

        sgl_idx = np.where(np.sum(grp_imgs, axis=0) > 0)

        plt.figure()
        plt.title("Mean Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(mean_img, vmin=0, vmax=1)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "mean_img_c" + str(grp) + "-" + measure_name + ".png"), mean_img, vmin=0, vmax=1)
        

        plt.figure()
        plt.title("Median Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(median_img, vmin=0, vmax=1)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "median_img_c" + str(grp) + "-" + measure_name + ".png"), median_img, vmin=0, vmax=1)
        
        plt.figure()
        plt.title("Max Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(max_img, vmin=0, vmax=1)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "max_img_c" + str(grp) + "-" + measure_name + ".png"), max_img, vmin=0, vmax=1)

        plt.figure()
        plt.title("Std Aggregation Image (Cluster %d)" % grp)
        #plt.imshow(std_img, vmin=0, vmax=0.5)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "std_img_c" + str(grp) + "-" + measure_name + ".png"), std_img, vmin=0, vmax=0.5)

        plt.figure()
        plt.title("Aggregation Hull Abstraction (Cluster %d)" % grp)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "hull_img_c" + str(grp) + "-" + measure_name + ".png"), opening(closing(hull_img, disk(2)), disk(2)))

        plt.figure()
        plt.title("Summation Image (Cluster %d)" % grp)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "sum_img_c" + str(grp) + "-" + measure_name + ".png"), sum_img)

        plt.figure()
        plt.title("Maximally Stable Extremal Regions Abtraction Image (Cluster %d)" % grp)
        plt.imsave(os.path.join(savepath, "c" + str(grp), additional_images, "mser_img_c" + str(grp) + "-" + measure_name + ".png"), mser_img)

        plt.close("all")



# Scatterplots for each pair of measures
def scatterplot(listA, listB, labels, measure_nameX, measure_nameY, savepath):
    if not labels:
        labels = np.arange(0, squareform(listA).shape[0])

    fig, ax = plt.subplots()
    plt.title("Distance Scatterplot")
    plt.plot(listA, listB, "bo")
    plt.plot([0,1], [0,1], "r")
    plt.xlabel(measure_nameX)
    plt.ylabel(measure_nameY)

    for lbl in range(len(listA)):
        idx = np.triu_indices_from(np.zeros((len(labels), len(labels))), k=1)
        tuples = np.array(range(len(labels))) * np.ones(len(labels))[:,None]
        tuples = np.dstack((tuples.T, tuples))
        tuples = tuples[idx[0], idx[1], :]
        ##if len(labels) > 0:
        # Use annotate only for interactive exploration. It clutters too much to save.
        #ax.annotate((labels[int(tuples[lbl][0])], labels[int(tuples[lbl][1])]), (scatterlists[0][lbl], scatterlists[1][lbl]))
        ##else:
        ##    ax.annotate(tuples[lbl], (scatterlists[0][lbl], scatterlists[1][lbl]))

    plt.savefig(os.path.join(savepath, "scatterplot-" + measure_nameX + "_" + measure_nameY + ".png"))

    plt.close("all")
