import sys
sys.path.append('.')
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import dbscan, spectral_clustering, affinity_propagation
from sklearn.cluster import OPTICS as sklearnOPTICS
from Pycluster import kmedoids as kmedoids_em
from pyclustering.cluster.kmedoids import kmedoids as kmedoids_pam
from calc_sd_matrix import similarity_distance_switcher

class Cluster:#method_name, savepath
    def __init__(self, dmatrix):
        self.dmatrix = dmatrix
        self.labels = None

    # Prepare arrays containing the similarity measure for each pair of one cluster once.
    def prep_cluster_similarity_array(self):
        if isinstance(self.dmatrix, type(None)) or isinstance(self.labels, type(None)):
            raise ValueError("Perform a clustering before calling this method!")
            
        if (np.diag(np.around(self.dmatrix,7)) == 0).all():
            dmatrix = self.dmatrix
        elif (np.diag(np.around(self.dmatrix,7)) == 1).all():
            dmatrix = similarity_distance_switcher(self.dmatrix)
        else:
            raise ValueError("Diagonal needs to be 0 or 1, for distance or similarity matrix, respectively.")
        
        c_arrays = []
        for i in range(1, np.amax(self.labels)+1):
            idx = np.where(self.labels == i)[0]
            sub_dmatrix = dmatrix[idx[:, None], idx]
            c_arrays.append(squareform(sub_dmatrix))
        return c_arrays



class Hierarchical(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)
        self.nr_cluster = nr_cluster

    def perform(self):
        cond_dmatrix = squareform(self.dmatrix)
        Z = linkage(cond_dmatrix, method="average", optimal_ordering=True)
        if self.nr_cluster == "auto":
            self.labels = fcluster(Z, t=0.3, criterion="distance")
        else:    
            self.labels = fcluster(Z, t=self.nr_cluster, criterion="maxclust")
        return self.labels



class AffinityPropagation(Cluster):
    def __init__(self, dmatrix):
        super().__init__(dmatrix)

    def perform(self):
        _, cl_labels = affinity_propagation(self.dmatrix)
        self.labels = self.process_affinity_propagation_memb(cl_labels)
        return self.labels

    # Bring affinity propagation output on par with hierarchical clustering output
    def process_affinity_propagation_memb(self, labels):
        if 0 in labels:
            self.labels = labels + 1
        return self.labels



class kMedoidsEM(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)
        #self.dmatrix = dmatrix
        self.nr_cluster = nr_cluster
        self.kmedoids_em_init, self.seed = self.find_seed()

    def perform(self):
        cl_centroid, _, _ = kmedoids_em(self.dmatrix, self.nr_cluster, initialid=self.kmedoids_em_init)
        self.labels = self.process_kmedoids_em_memb(cl_centroid)
        return self.labels

    # Bring k-medoids-em output on par with hierarchical clustering output
    def process_kmedoids_em_memb(self, centroids):
        codes = {val: idx for idx, val in enumerate(set(centroids), start=1)}
        self.labels = np.array([codes[val] for val in centroids])
        return self.labels

    # Needed for determinism
    # Find a seed such that the required number of clusters is provided.
    def find_seed(self):
        seed_idx = -1
        while True:
            seed_idx += 1
            np.random.seed(seed_idx)
            kmedoids_em_init = np.random.randint(0, self.nr_cluster, size=self.dmatrix.shape[0])
            if len(set(kmedoids_em_init)) == self.nr_cluster:
                print("Random Seed for kMedoids EM: %i"%(seed_idx))
                break
        return kmedoids_em_init, seed_idx



class kMedoidsPAM(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)
        self.nr_cluster = nr_cluster
        self.kmedoids_pam_init, self.seed = self.find_seed()

    def perform(self):
        cl_labels = kmedoids_pam(self.dmatrix, self.kmedoids_pam_init, data_type="distance_matrix").process().get_clusters()
        self.labels = self.process_kmedoids_pam_memb(cl_labels).astype(int)
        return self.labels

    # Bring k-medoids-pam output on par with hierarchical clustering output
    def process_kmedoids_pam_memb(self, membership_lists):
        max_idx = np.amax([idx for memb_list in membership_lists for idx in memb_list])
        pam_labels = np.zeros(max_idx + 1)
        for memb_idx, memb_list in enumerate(membership_lists, start=1):
            for idx in memb_list:
                pam_labels[idx] = memb_idx
        self.labels = pam_labels
        return self.labels

    # Needed for determinism
    # Find a seed such that the required number of clusters is provided.
    def find_seed(self):
        seed_idx = -1
        while True:
            seed_idx += 1
            np.random.seed(seed_idx)
            kmedoids_pam_init = np.random.randint(0, self.dmatrix.shape[0]-1, size=int(self.nr_cluster))
            if len(set(kmedoids_pam_init)) == self.nr_cluster:
                print("Random Seed: %i"%(seed_idx))
                break
        return kmedoids_pam_init, seed_idx



class DBSCAN(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)
        p99 = np.percentile(self.dmatrix[np.triu_indices_from(self.dmatrix, k=1)], 99)
        p1 = np.percentile(self.dmatrix[np.triu_indices_from(self.dmatrix, k=1)], 1)
        self.sample_eps2 = (p99+p1) * 0.3

    def perform(self):
        core_samples, cl_labels = dbscan(self.dmatrix, eps=self.sample_eps2, min_samples=2, metric="precomputed")
        self.labels = self.process_dbscan_memb(cl_labels)
        return self.labels

    # Bring dbscan output on par with hierarchical clustering output
    def process_dbscan_memb(self, labels):
        labels = np.array(labels)
        noise_idx = np.where(labels==-1)[0]
        for label, noise_idx in enumerate(noise_idx, start=np.amax(labels)+1):
            labels[noise_idx] = label
        self.labels = labels+1
        return self.labels



class OPTICS(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)

    def perform(self):
        cl_labels = sklearnOPTICS(min_samples=2, xi=0.001, metric="precomputed").fit_predict(self.dmatrix)
        self.labels = self.process_optics_memb(cl_labels)
        return self.labels

    # Bring optics output on par with hierarchical clustering output
    def process_optics_memb(self, labels):
        labels = np.array(labels)
        noise_idx = np.where(labels==-1)[0]
        for label, noise_idx in enumerate(noise_idx, start=np.amax(labels)+1):
            labels[noise_idx] = label
        self.labels = labels+1
        return self.labels



class Spectral(Cluster):
    def __init__(self, dmatrix, nr_cluster):
        super().__init__(dmatrix)
        p99 = np.percentile(self.dmatrix[np.triu_indices_from(self.dmatrix, k=1)], 99)
        p1 = np.percentile(self.dmatrix[np.triu_indices_from(self.dmatrix, k=1)], 1)
        self.delta = p99-p1

    def perform(self):
        cl_labels = spectral_clustering(np.exp(- self.dmatrix ** 2 / (2. * self.delta ** 2)), n_clusters=self.nr_cluster, random_state=0, assign_labels="discretize")
        self.labels = self.process_spectral_clustering_memb(cl_labels) 
        return self.labels

    # Bring spectral clustering output on par with hierarchical clustering output
    def process_spectral_clustering_memb(self, labels):
        self.labels = labels+1
        return self.labels