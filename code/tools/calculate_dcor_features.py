import numpy as np
from base.classifiers import PairwiseContinuous
import itertools
import multiprocessing
from base.statistics import dcor
from base import tools
from scipy.stats import pearsonr

def parallel_dcor(args):
    (fi, other_regions), region = args
        
    other_regions.remove(region)
    results = []
    for combs in itertools.combinations(other_regions, 2):
        results.append(dcor(fi[region, combs[0]], fi[region, combs[1]]))
        
    return np.array(results).mean()

def parallel_pearson(args):
    (fi, other_regions), region = args
        
    other_regions.remove(region)
    results = []
    for combs in itertools.combinations(other_regions, 2):
        results.append(pearsonr(fi[region, combs[0]], fi[region, combs[1]])[0])
        
    return np.array(results).mean()

def calc_dcor_features(pairwise_clf, processes=7, statistic = 'dcor'):

	pool = multiprocessing.Pool(processes)

	regions = range(0, pairwise_clf.mask_num)
	results = []

	pb = tools.ProgressBar(len(regions), start=True)

	if statistic == 'dcor':
	 	parallel_func = parallel_dcor
	else:
		parallel_func = parallel_pearson

	for output in pool.imap(parallel_func, itertools.izip(itertools.repeat((pairwise_clf.feature_importances, regions)), regions)):
		results.append(output)
		pb.next()

	return results

infile = '../results/ward_f60_Pairwise_Ridge_rz_abs_topics60_filt/classifier.pkl'

pairwise_clf = PairwiseContinuous.load(infile)
pairwise_clf.dcor_features = calc_dcor_features(pairwise_clf)
pairwise_clf.save(infile)