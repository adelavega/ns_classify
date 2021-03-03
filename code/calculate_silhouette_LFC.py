
from sklearn.metrics import silhouette_score
import joblib
from scipy.cluster.hierarchy import fcluster
import pandas as pd
import numpy as np
import nibabel as nib
from neurosynth.analysis.cluster import Clusterable
from neurosynth.base.dataset import Dataset
from sklearn.metrics import adjusted_rand_score

dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')

distances = joblib.load('/home/delavega/projects/clustering/results/hierarchical/v6_distances_scaled.pkl')
Z = joblib.load('/home/delavega/projects/clustering/results/hierarchical/v6_ward_c30_scaled.pkl')
wb_Z = joblib.load('/home/delavega/projects/clustering/results/hierarchical/v6_ward_c30_scaled.pkl')
LFC_mask = nib.load('/home/delavega/projects/classification/masks/LFC_MNI_noMedialOFC.nii.gz')

whole_brain_masker = Clusterable(dataset, '/home/delavega/projects/classification/masks/HO_ROIs/cortex_30.nii.gz', min_studies=100).masker

sils = []
masked_labels = []
for i, n_reg in enumerate(np.arange(40, 120)):
	last_labels = masked_labels
	labels = fcluster(wb_Z, n_reg, 'maxclust')
	masked_dist = distances[[whole_brain_masker.mask(LFC_mask.get_data()) == 1]]
	masked_labels = labels[whole_brain_masker.mask(LFC_mask.get_data()) == 1]
	if (i > 0) and (adjusted_rand_score(masked_labels, last_labels) != 1):
	    sil = silhouette_score(masked_dist, masked_labels)
	    sils.append([n_reg, sil])   
	    pd.DataFrame(sils, columns=['n_clusters', 'silhouette_score']).to_csv(
	    	'/home/delavega/projects/clustering/results/hierarchical/v6_ward_c30_scaled_LFC_sils_40_120.csv')
