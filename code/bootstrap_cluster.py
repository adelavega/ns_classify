from neurosynth.base.dataset import Dataset
from neurosynth.analysis.cluster import Clusterable
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pickle
from sklearn.metrics import silhouette_score
import pandas as pd
import sys
from sklearn.utils import resample
from joblib import Parallel, delayed

try:
	job, n_min, n_max, iterations, job_id = sys.argv
except:
	raise Exception("Incorect number of arguments")

n_min = int(n_min)
n_max = int(n_max)
iterations = int(iterations)
job_id = int(job_id)

dataset = Dataset.load('/home/delavega/projects/classification/data/datasets/abs_60topics_filt_jul.pkl') ## Replace

infile = '/home/delavega/projects/classification/masks/HO_ROIs/new_medial_fc_30.nii.gz'
saved_pca = '/projects/delavega/clustering/dv_v5_reference_min_80_pca.pkl'

print "Loading saved PCAs"
reference = pickle.load(open(saved_pca, 'r'))

roi = Clusterable(dataset, infile, min_studies=80)

silhouettes = []

def cluster_parallel(X, Y, n_clusters, i):
	from numpy.random import seed
	seed(i)
	
	X, Y = resample(roi.data.T, reference.data.T)

	print "Calculating distance"
	distances = pairwise_distances(X.T, Y.T, metric='correlation')

	clustering_algorithm = KMeans(n_clusters = n_clusters)
	clustering_algorithm = clustering_algorithm.fit(distances) 

	labels = clustering_algorithm.predict(distances)

	return silhouette_score(distances, labels)

for n_clusters in range(n_min, n_max + 1):
	silhouettes = Parallel(n_jobs=1)(delayed(cluster_parallel)(
		roi.data, reference.data, n_clusters, i) for i in range(iterations))
	save_sil = pd.DataFrame(silhouettes, columns=['sil'])
	save_sil['n_clusters'] = n_clusters
	save_sil.to_csv('/home/delavega/projects/clustering/results/bootstrap_silhouettes/MFC_boot_%d_%d_%d.csv' % (n_clusters, iterations, job_id), index=False)

