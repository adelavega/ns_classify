from neurosynth.base.dataset import Dataset
from neurosynth.analysis.cluster import Clusterable

from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
from sklearn.preprocessing import scale

from fastcluster import ward

from joblib import Parallel, delayed, dump, load

def cluster_ward(roi, reference, i):
	# from numpy.random import seed
	# seed(i)
	# X, Y = resample(roi.T, reference.T)

	# print "Computing roi ref distances..."
	# distances = pairwise_distances(X.T, Y.T, metric='correlation')
	# scaled_distances = scale(distances, axis=1)

	try:
		distances = load('/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_b%d.pkl' % i)
		scaled_distances = load('/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_scaled_b%d.pkl' % i)

	# dump(distances, '/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_b%d.pkl' % i)
	# dump(scaled_distances, '/projects/delavega/clustering/results/bootstrap/hierarchical/whole_brain_PCA_dist_min100_scaled_b%d.pkl' % i)

		Z = ward(distances.T)
		Z_scaled = ward(scaled_distances.T)

		dump(Z, '/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_b%d.pkl' % i)
		dump(Z_scaled, '/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_scaled_b%d.pkl' % i)
	except IOError:
		pass


# dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')
# saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
# reference = load(saved_pca)

n_boot = 100

# roi = Clusterable(dataset, min_studies=100)

Parallel(n_jobs=2)(delayed(cluster_ward)(
    'roi.data', 'reference.data', i) for i in range(n_boot))
