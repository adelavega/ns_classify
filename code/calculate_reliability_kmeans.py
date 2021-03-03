import glob
from joblib import Parallel, delayed, load
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd

def reproducibility_rating(labels, type='ars'):
    """ Run mutliple pairwise supervised ratings to obtain an average
    rating
    Parameters
    ----------
    labels: list of label vectors to be compared
    Returns
    -------
    av_score:  float, a scalar summarizing the reproducibility of pairs of maps
    """

    if type == 'ars':
    	scorer = adjusted_rand_score
    else:
    	scorer = adjusted_mutual_info_score

    av_score = 0
    niter = len(labels) 
    for i in range(1, niter):
        for j in range(i):
            av_score += scorer(labels[j], labels[i])
            av_score += scorer(labels[i], labels[j])
    av_score /= (niter * (niter - 1))

    return av_score

def reproducibility_across_regions(all_labels, n_reg):
    return [reproducibility_rating(all_labels), reproducibility_rating(all_labels, type='ami'), n_reg]


all_labels = [load(f) for f in glob.glob('/home/delavega/projects/clustering/results/bootstrap/kmeans/labels200*scaled*.pkl')]
by_regions = [[r for boot in all_labels for r in boot if r[0] == reg] for reg in zip(*all_labels[0])[0]]

results = Parallel(n_jobs=8)(delayed(reproducibility_across_regions)(zip(*reg)[1], zip(*reg)[0][0]) for reg in by_regions)

results_scaled = pd.DataFrame(results, columns=['ars', 'ami', 'n'])
results_scaled['type'] = 'scaled'

all_labels = [load(f) for f in glob.glob('/home/delavega/projects/clustering/results/bootstrap/kmeans/labels200*raw*.pkl')]
by_regions = [[r for boot in all_labels for r in boot if r[0] == reg] for reg in zip(*all_labels[0])[0]]

results = Parallel(n_jobs=8)(delayed(reproducibility_across_regions)(zip(*reg)[1], zip(*reg)[0][0]) for reg in by_regions)	

results = pd.DataFrame(results, columns=['ars', 'ami', 'n'])
results['type'] = 'raw'

all_results = pd.concat([results_scaled, results])
all_results['algorithm'] = 'kmeans'

all_results.to_csv('/home/delavega/projects/clustering/results/bootstrap/hierarchical/ns_kmeans_reliability_180.csv')
