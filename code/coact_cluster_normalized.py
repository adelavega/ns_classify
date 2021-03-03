from neurosynth.base.dataset import Dataset
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale
from neurosynth.analysis.cluster import Clusterable
dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')
from fastcluster import ward

roi = Clusterable(dataset, '/home/delavega/projects/classification/masks/l_70_mask.nii.gz')

saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
reference = joblib.load(saved_pca)

distances = pairwise_distances(roi.data, reference.data, metric='correlation')
distances = scale(distances, axis=1)

joblib.dump(distances, '/home/delavega/projects/clustering/results/hierarchical/v6_distances_l_70_scaled.pkl')

Z = ward(distances)

joblib.dump(Z, '/home/delavega/projects/clustering/results/hierarchical/v6_ward_l70_scaled.pkl')