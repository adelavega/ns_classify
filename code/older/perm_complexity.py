from sklearn.metrics import roc_auc_score
import sys
from base.mv import bootstrap_mv_full
from neurosynth.base.dataset import Dataset
dataset = Dataset.load("../permutation_clustering/abs_60topics_filt_jul.pkl")

from sklearn.linear_model import LassoLarsIC

print sys.argv
try:
	cmd, iterations, job_id = sys.argv
except:
	raise Exception("Incorect number of arguments")

import csv
cognitive_topics = ['topic' + topic[0] for topic in csv.reader(open('topic_keys60-july_cognitive.csv', 'rU')) if topic[1] == "T"]

results = bootstrap_mv_full(dataset, LassoLarsIC(), roc_auc_score, 
	'../permutation_clustering/results/medial_fc_30_kmeans/kmeans_k9/cluster_labels.nii.gz', features=cognitive_topics, processes=None, 
	boot_n=int(iterations), outfile='results/bootstrap_full_mv_' + str(iterations) + '_mFC__LASSO_LARS_60_ ' + str(job_id) + '.csv')

