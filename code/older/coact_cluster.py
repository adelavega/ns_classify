#!/usr/bin/env python
from neurosynth.base.dataset import Dataset
import sys
from sklearn import decomposition as sk_decomp
from os.path import exists, join
from os import makedirs
from neurosynth.analysis.cluster import Clusterable
from sklearn.metrics import pairwise_distances
from nibabel import nifti1
from sklearn.cluster import KMeans
# from mlpy import MFastHCluster
import pickle
import numpy as np

try:
	cmd, min_n, max_n, step = sys.argv
except:
	raise Exception("Incorect number of arguments")

# class FastWard(object):
#     def __init__(self):
#         pass
    
#     def fit(self, X):
#         self.cf = MFastHCluster(method='ward')
#         self.cf.linkage(X)
        
#     def predict(self, n):
#         for i in range(1, self.cf.cut(0).shape[0]):
#             labels = self.cf.cut(i)
#             if np.bincount(labels).shape[0] == n:
#                 break
                
#         return labels

mydir = "/projects/delavega/clustering/"
dataset = Dataset.load(mydir + 'abs_60topics_filt_jul.pkl')

roi_mask = mydir + 'masks/new_medial_fc_30.nii.gz'
ns =  [3, 9]
save_images = True
output_dir = join(mydir, 'results/MFC/')
out_model = None

roi = Clusterable(dataset, roi_mask, min_studies=80)

reference = Clusterable(dataset, min_studies=80)
reduce_reference = sk_decomp.RandomizedPCA(100)
reference = reference.transform(reduce_reference, transpose=True)

# distances = pairwise_distances(roi.data, reference.data,
#                                metric='correlation')

# clustering_algorithm.fit(distances)

reference_data = reference.data
roi_data = roi.data

# if out_model is not None:
#     pickle.dump(clustering_algorithm, open(out_model, 'wb'), -1)

for n_clusters in ns:

    ## Bootstrap


    clustering_algorithm = KMeans(n_clusters = n_clusters)

    for boot in range(0, 100):

        ## Randomly sample data & recalculate distances
        ran_index = np.random.choice(reference_data.shape[1], reference_data.shape[1])
        new_reference_data = reference_data[:, ran_index]
        new_roi_data = roi_data[:, ran_index]

        distances = pairwise_distances(new_roi_data, new_reference_data,
                               metric='correlation')

        clustering_algorithm.fit(distances) 

        labels = clustering_algorithm.predict(distances) + 1
        
        # Make nibabel image
        header = roi.masker.get_header()
        header['cal_max'] = labels.max()
        header['cal_min'] = labels.min()
        voxel_labels = roi.masker.unmask(labels)
        img = nifti1.Nifti1Image(voxel_labels, None, header)

        if save_images:
            sub_dir = join(output_dir, 'kmeans_' + str(boot) + '_' + str(n_clusters))
            if not exists(sub_dir):
                makedirs(sub_dir)

            filename = 'cluster_labels_k%d.nii.gz' % n_clusters
            outfile = join(sub_dir, filename)
            
            img.to_filename(outfile)