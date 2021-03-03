from neurosynth.base.dataset import Dataset
from neurosynth.analysis.cluster import Clusterable

from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
from sklearn.preprocessing import scale

from fastcluster import ward

from joblib import Parallel, delayed, dump, load

import numpy as np
from sklearn.cluster import KMeans

def cluster_kmeans(roi, reference, regions, i):
	# from 7numpy.random import seed
	# seed(i)
	# X, Y = resample(roi.T, reference.T)

	# print "Computing roi ref distances..."
	# distances = pairwise_distances(X.T, Y.T, metric='correlation')
	# scaled_distances = scale(distances, axis=1)

	# Load distances

	try:
		distances = load('/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_b%d.pkl' % i).T
		scaled_distances = load('/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_scaled_b%d.pkl' % i).T

		all_raw = []
		for n_clusters in regions:
		    clustering_algorithm = KMeans(n_clusters = n_clusters)
		    clustering_algorithm.fit(distances) 

		    all_raw.append([n_clusters, clustering_algorithm.predict(distances)])

		dump(all_raw, '/projects/delavega/clustering/results/bootstrap/kmeans/labels200_wholebrain_raw_b%d.pkl' % i)

		all_scaled = []
		for n_clusters in regions:
		    clustering_algorithm = KMeans(n_clusters = n_clusters)
		    clustering_algorithm.fit(scaled_distances) 

		    all_scaled.append([n_clusters, clustering_algorithm.predict(scaled_distances)])

		dump(all_scaled, '/projects/delavega/clustering/results/bootstrap/kmeans/labels200_wholebrain_scaled_b%d.pkl' % i)
	except IOError:
		pass


# dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')
# saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
# reference = load(saved_pca)


# roi = Clusterable(dataset, min_studies=100)

n_boot = (1, 25)
# n_boot = (11, 15)
# n_boot = (15, 20)
# n_boot = (51, 55)
n_boot = (25, 50)
# n_boot = (27, 30)

# n_boot = (25, 27)
# n_boot = (27, 30)
# n_boot = (32, 34)
# n_boot = (34, 36)
# n_boot = (36, 38)
# n_boot = (38, 40)
# n_boot = (41, 43)
# n_boot = (43, 45)

# n_boot = (55, 60)
# n_boot = (60, 65)
# n_boot = (65, 70)
# n_boot = (70, 75)
# n_boot = (75, 80)
# n_boot = (80, 85)
# n_boot = (85, 90)
# n_boot = (95, 100)

regions = [180]

Parallel(n_jobs=1)(delayed(cluster_kmeans)('reference.data', 'roi.data', regions, i) for i in range(*n_boot))
