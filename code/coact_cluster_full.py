from neurosynth.base.dataset import Dataset
from neurosynth.analysis.cluster import Clusterable

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale
from fastcluster import ward
from joblib import dump, load, Parallel, delayed
import numpy as np
from sklearn.cluster import KMeans


def cluster_kmeans(distances, scaled_distances, regions):
	all_raw = []
	for n_clusters in regions:
	    clustering_algorithm = KMeans(n_clusters = n_clusters)
	    clustering_algorithm.fit(distances) 

	    all_raw.append([n_clusters, clustering_algorithm.predict(distances)])

	dump(all_raw, '/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_raw_full_180.pkl')

	all_scaled = []
	for n_clusters in regions:
	    clustering_algorithm = KMeans(n_clusters = n_clusters)
	    clustering_algorithm.fit(scaled_distances) 

	    all_scaled.append([n_clusters, clustering_algorithm.predict(scaled_distances)])

	dump(all_scaled, '/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_scaled_full_180.pkl')

def cluster_ward(distances, scaled_distances):
	Z = ward(distances)
	Z_scaled = ward(scaled_distances)

	dump(Z, '/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_full.pkl')
	dump(Z_scaled, '/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_scaled_full.pkl')


dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')
saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
reference = load(saved_pca)
roi = Clusterable(dataset, min_studies=100)

# distances = pairwise_distances(roi.data, reference.data, metric='correlation')
# scaled_distances = scale(distances, axis=1)
distances = load('/projects/delavega/clustering/results/bootstrap/whole_brain_PCA_dist_min100_full.kl')
scaled_distances = load('/projects/delavega/clustering/results/bootstrap/whole_brain_PCA_dist_min100_scaled_full.pkl')

cluster_kmeans(distances, scaled_distances, [180])
