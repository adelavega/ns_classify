from neurosynth.base.dataset import Dataset
from neurosynth.analysis.cluster import Clusterable
from sklearn import decomposition as sk_decomp
import pickle

dataset = Dataset.load('/home/delavega/projects/classification/data/datasets/abs_60topics_filt_jul.pkl')

out = '/projects/delavega/clustering/dv_v5_reference_min_80_pca.pkl'

reference = Clusterable(dataset,min_studies=80)
print "Running PCA"
reduce_reference = sk_decomp.RandomizedPCA(100)
reference = reference.transform(reduce_reference, transpose=True)

pickle.dump(reference, open(out, 'w'))