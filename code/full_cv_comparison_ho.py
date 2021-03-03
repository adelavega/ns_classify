import nibabel as nib
import numpy as np
from neurosynth.base.dataset import Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

from neurosynth.analysis.cluster import Clusterable
from neurosynth.analysis.reduce import average_within_regions
from joblib import Parallel, delayed
from nibabel import nifti1
from copy import deepcopy
import pandas as pd

dataset = Dataset.load('/projects/delavega/dbs/db_v6_topics-100.pkl')

def binarize_nib(img):
    img = deepcopy(img)
    img.get_data()[img.get_data() != 0] = 1
    return img

def cv_distances(dataset, reference, mask, train_index):
    roi = Clusterable(dataset, mask)
    roi.data = roi.data[:, train_index]

    reference = reference.data[:, train_index]

    print "Computing roi ref distances..."
    distances = pairwise_distances(roi.data, reference, metric='correlation')
    distances[np.isnan(distances)] = 1
    return distances, roi

def cluster(dataset, distances, roi, n_clusters):
    print "Clustering: " + str(n_clusters)
    clustering_algorithm = KMeans(n_clusters = n_clusters)
    clustering_algorithm.fit(distances) 

    labels = clustering_algorithm.predict(distances) + 1
    
    ### Try shortening this
    header = dataset.masker.get_header()
    header['cal_max'] = labels.max()
    header['cal_min'] = labels.min()
    voxel_labels = roi.masker.unmask(labels)
    img = nifti1.Nifti1Image(voxel_labels, None, header)
        
    return img

def fit_predict(classifier, X_train, X_test, y_train, y_test):
    print "Classifying"
    classifier.fit(X_train, y_train)
    return [y_test, classifier.predict(X_test).tolist()]

saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
reference = pickle.load(open(saved_pca, 'r'))
classifier = GaussianNB()
cver = KFold(dataset.feature_table.data.shape[0], n_folds = 4)

name = 'h_o_asym'
from nilearn.datasets import fetch_atlas_harvard_oxford

match_region = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True).maps
n_regions = np.unique(match_region.get_data()).nonzero()[0].shape[0]

distances = Parallel(n_jobs=1)(delayed(cv_distances)(
    dataset, reference, binarize_nib(match_region), train_index) for train_index, _ in cver)

print "Getting ready to cluster"
ns_matched_regions = Parallel(n_jobs=-1)(delayed(cluster)(dataset, d, r, n_regions) for d, r in distances)

print "Classifying"
all_predictions = []
### For each fold, predict activation with topics using corresponding clustering
for fold_i, (train_index, test_index) in enumerate(cver):
    ys = (dataset.feature_table.data.values > 0.001).astype('int').T

    X = average_within_regions(dataset, ns_matched_regions[fold_i]).T
    match_predictions = Parallel(n_jobs=-1)(delayed(fit_predict)(
        classifier, X[train_index, :], X[test_index, :], y[train_index], y[test_index]) for y in ys)
    match_predictions = pd.DataFrame(match_predictions, columns=['y_test', 'y_pred'])
    match_predictions['region'] = dataset.get_feature_names()
    match_predictions['atlas'] = 'ns_%s' % name
    match_predictions['fold'] = fold_i
    match_predictions['n_regions'] = n_regions
    all_predictions.append(match_predictions)

    print "Gordon"
    X = average_within_regions(dataset, match_region).T
    match_predictions = Parallel(n_jobs=-1)(delayed(fit_predict)(
        classifier, X[train_index, :], X[test_index, :], y[train_index], y[test_index]) for y in ys)
    match_predictions = pd.DataFrame(match_predictions, columns=['y_test', 'y_pred'])
    match_predictions['region'] = dataset.get_feature_names()
    match_predictions['atlas'] = name
    match_predictions['fold'] = fold_i
    match_predictions['n_regions'] = n_regions
    all_predictions.append(match_predictions)


all_predictions = pd.concat(all_predictions)
all_predictions.to_pickle('/home/delavega/projects/classification/results/cv_clust_predict/topics_%s_%d.pkl' % (name, n_regions))