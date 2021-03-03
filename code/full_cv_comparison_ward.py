import nibabel as nib
import numpy as np
from neurosynth.base.dataset import Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
import pickle
from sklearn.metrics import pairwise_distances

from neurosynth.analysis.cluster import Clusterable
from neurosynth.analysis.reduce import average_within_regions
from joblib import Parallel, delayed
from nibabel import nifti1
from copy import deepcopy
import pandas as pd

dataset = Dataset.load('../data/0.6/datasets/db_v6_topics-100.pkl')

from mlpy import MFastHCluster

class FastWard(object):
    def __init__(self):
        pass
    
    def fit(self, X):
        self.cf = MFastHCluster(method='ward')
        self.cf.linkage(X)
        
    def predict(self, n, step=0.01):
        for i in np.arange(1, self.cf.cut(0).shape[0], step):
            labels = self.cf.cut(i)
            if np.bincount(labels).shape[0] <= n:
                break
                
        return labels

def binarize_nib(img):
    img = deepcopy(img)
    img.get_data()[img.get_data() != 0] = 1
    return img

def cv_cluster_ward(dataset, reference, mask_img, train_index, n_regions):
    mask_img = binarize_nib(mask_img)
    roi = Clusterable(dataset, mask_img)
    roi.data = roi.data[:, train_index]

    print "Computing roi ref distances..."
    distances = pairwise_distances(roi.data, reference, metric='correlation')
    distances[np.isnan(distances)] = 1

    clustering_algorithm = FastWard()
    clustering_algorithm.fit(distances) 

    imgs = []

    for region in n_regions:
        labels = clustering_algorithm.predict(region, step=0.01) + 1
        header = dataset.masker.get_header()
        header['cal_max'] = labels.max()
        header['cal_min'] = labels.min()
        voxel_labels = roi.masker.unmask(labels)
        imgs.append(nifti1.Nifti1Image(voxel_labels, None, header))
    
    return imgs

def fit_predict(classifier, X_train, X_test, y_train, y_test):
    print "Classifying"
    classifier.fit(X_train, y_train)
    return [y_test, classifier.predict(X_test).tolist()]

saved_pca = '../results/clustering/dv_v6_reference_pca.pkl'
reference = pickle.load(open(saved_pca, 'r'))
classifier = GaussianNB()
cver = KFold(dataset.feature_table.data.shape[0], n_folds = 4)

match_region = nib.load('../masks/craddock/scorr_05_2level/%d.nii.gz' % 100)

regions = [20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 129, 140, 150, 160, 169, 180, 190, 199, 209, 218, 229, 239]

### Generate matched clustering for each fold of ns data
ns_regions = Parallel(n_jobs=1)(delayed(cv_cluster_ward)(
    dataset, reference.data[:, train_index], match_region, train_index, regions) for train_index, _ in cver)

name = 'ns_craddock_ward'
### For number of regions
for n_ix, _ in enumerate(regions):
    ns_matched_regions = [fold[n_ix] for fold in ns_regions]
    n_regions  = np.unique(ns_matched_regions[0].get_data()).shape[0]
    all_predictions = []
    ### For each fold, predict activation with topics using corresponding clustering
    for fold_i, (train_index, test_index) in enumerate(cver):
        ys = (dataset.feature_table.data.values > 0.001).astype('int').T

        X = average_within_regions(dataset, ns_matched_regions[fold_i]).T
        match_predictions = Parallel(n_jobs=7)(delayed(fit_predict)(
            classifier, X[train_index, :], X[test_index, :], y[train_index], y[test_index]) for y in ys)
        match_predictions = pd.DataFrame(match_predictions, columns=['y_test', 'y_pred'])
        match_predictions['region'] = dataset.get_feature_names()
        match_predictions['atlas'] = name
        match_predictions['fold'] = fold_i
        match_predictions['n_regions'] = np.unique(ns_matched_regions[fold_i].get_data()).shape[0]

        all_predictions.append(match_predictions)

    all_predictions = pd.concat(all_predictions)
    all_predictions.to_pickle('../results/classification/cv_clust_predict/topics_%s_%d.pkl' % (name, n_regions))