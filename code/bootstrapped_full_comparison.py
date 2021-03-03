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

def cv_cluster(dataset, reference, mask_img, train_index):
    n_clusters = np.unique(mask_img.get_data()).nonzero()[0].shape[0]
    mask_img = binarize_nib(mask_img)
    roi = Clusterable(dataset, mask_img)
    roi.data = roi.data[:, train_index]

    print "Computing roi ref distances..."
    distances = pairwise_distances(roi.data, reference, metric='correlation')
 
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
    classifier.fit(X_train, y_train)
    return [y_test, classifier.predict(X_test).tolist()]

 ### For number of regions
match_regions = nib.load('masks/craddock/scorr_05_2level/20.nii.gz')
saved_pca = '/projects/delavega/clustering/dv_v6_reference_pca.pkl'
reference = pickle.load(open(saved_pca, 'r'))

classifier = GaussianNB()
cver = KFold(dataset.feature_table.data.shape[0], n_folds=4)
predictions = []
n_regions = int(match_regions.get_data().max())

### Generate matched clustering for each fold of ns data
ns_regions = Parallel(n_jobs=-1)(delayed(cv_cluster)(
    dataset, reference.data[:, train_index], match_regions, train_index) for train_index, _ in cver)

all_predictions = []
### For each fold, predict activation with topics using corresponding clustering
for fold_i, (train_index, test_index) in enumerate(cver):
    X_train = dataset.feature_table.data.iloc[train_index, :]
    X_test = dataset.feature_table.data.iloc[test_index, :]

    ## Predict using matched neurosynth atlas
    ys = average_within_regions(dataset, ns_regions[fold_i], threshold=0.05).astype('int64')
    ns_predictions = Parallel(n_jobs=-1)(delayed(fit_predict)(
		classifier, X_train, X_test, y[train_index], y[test_index]) for y in ys)
    ns_predictions = pd.DataFrame(ns_predictions, columns=['y_test', 'y_pred'])
    ns_predictions['region'] = range(0, n_regions + 1)
    ns_predictions['atlas'] = 'ns_craddock'
    ns_predictions['fold'] = fold_i
    all_predictions.append(ns_predictions)

    ## Predict using original craddock atlas
    ys = average_within_regions(dataset, match_regions, threshold=0.05).astype('int64')
    match_predictions = Parallel(n_jobs=-1)(delayed(fit_predict)(
        classifier, X_train, X_test, y[train_index], y[test_index]) for y in ys)
    match_predictions = pd.DataFrame(match_predictions, columns=['y_test', 'y_pred'])
    match_predictions['region'] = range(0, n_regions + 1)
    match_predictions['atlas'] = 'craddock'
    match_predictions['fold'] = fold_i
    all_predictions.append(match_predictions)

all_predictions = pd.concat(all_predictions)
all_predictions.save('/home/delavega/projects/classification/results/cv_clust_predict/craddock_20.pkl')
