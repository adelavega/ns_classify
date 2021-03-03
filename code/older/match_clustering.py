from neurosynth.base.dataset import Dataset
import nibabel as nib
import numpy as np
from neurosynth.analysis.cluster import Clusterable
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from nibabel import nifti1
from copy import deepcopy
from os.path import exists, join, split
from os import makedirs
import pickle

def binarize_nib(img):
    img = deepcopy(img)
    img.get_data()[img.get_data() != 0] = 1
    return img

dataset = Dataset.load('../data/0.6/datasets/db_v6_topics-100.pkl')

infile = '../masks/mars/NeubertCingulateOrbitoFrontalParcellation/Neubert_2mm_medial_bilateral.nii.gz'
outdir = '../results/clustering/matched/mars/'
outfile = join(outdir, split(infile)[-1])
saved_pca = '../results/clustering/dv_v6_reference_pca.pkl'

reference = pickle.load(open(saved_pca, 'r'))

match_roi = nib.load(infile)
roi = Clusterable(dataset, binarize_nib(match_roi))

print "Computing roi ref distances"
distances = pairwise_distances(roi.data, reference.data, metric='correlation')

n_clusters = np.unique(match_roi.get_data()).nonzero()[0].shape[0]

print "Clustering: " + str(n_clusters)
clustering_algorithm = KMeans(n_clusters = n_clusters)
clustering_algorithm.fit(distances) 

labels = clustering_algorithm.predict(distances) + 1

# Make nibabel image
header = roi.masker.get_header()
header['cal_max'] = labels.max()
header['cal_min'] = labels.min()
voxel_labels = roi.masker.unmask(labels)
img = nifti1.Nifti1Image(voxel_labels, None, header)

if not exists(outdir):
	makedirs(outdir)

img.to_filename(outfile)
