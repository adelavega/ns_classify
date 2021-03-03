from base.classifiers import OnevsallContinuous
from neurosynth.analysis import cluster
from neurosynth.base.dataset import Dataset
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import cPickle
from sklearn.decomposition import RandomizedPCA

dataset = Dataset.load('../data/datasets/abs_60topics_filt_jul.pkl')

roi_mask = '../masks/mpfc_nfp.nii.gz'
global_mask = "../masks/MNI152_T1_2mm_brain.nii.gz"

n_regions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# print resolution
clf_file = "../data/clfs/all_vox_Ridge_mpfc.pkl"

print "trying to load"
try: 
	clf = OnevsallContinuous.load(clf_file)
except:
	print "Loading failed"
	clf = OnevsallContinuous(dataset, None, classifier = Ridge())
	clf.classify(scoring=r2_score, processes=8)
	try:
		clf.save(clf_file)
	except:
		pass

reduc = RandomizedPCA(n_components=100)

print "Setting up clustering"
clstr = cluster.Clusterer(dataset, 'coactivation', global_mask=global_mask, 
    output_dir='../results/cluster/coact_PCA_mpfc/', 
    roi_mask=roi_mask, min_studies_per_voxel=25, voxel_parcellation=reduc,
    distance_metric='correlation', semantic_data = clf.feature_importances)

print "Clustering..."
sil = clstr.cluster(algorithm='kmeans', n_clusters=n_regions, bundle=False, coactivation_maps=False, precomputed_distances=True)

cPickle.dump(open('../results/cluster/coact_PCA_mpfc/sil.pkl', 'wb'))
