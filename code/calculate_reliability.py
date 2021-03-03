import glob
from joblib import Parallel, delayed, load
import numpy as np
from scipy.cluster.hierarchy import fcluster
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

def reproducibility_across_regions(all_labels, i, n_reg):
    return [reproducibility_rating(zip(*all_labels)[i]), n_reg]


regions = np.arange(20, 1020, 20)

Zs = [load(f) for f in glob.glob('/home/delavega/projects/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_scaled_b*.pkl')]
all_labels = [Parallel(n_jobs=32)(delayed(fcluster)(Z, region, 'maxclust') for region in regions) for Z in Zs]

results = Parallel(n_jobs=8)(delayed(reproducibility_across_regions)(all_labels, i, n_reg) for i, n_reg in enumerate(regions))

results_scaled = pd.DataFrame(results, columns=['ars', 'n'])
results_scaled['type'] = 'scaled'

Zs = [load(f) for f in glob.glob('/home/delavega/projects/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_b*.pkl')]
all_labels = [Parallel(n_jobs=32)(delayed(fcluster)(Z, region, 'maxclust') for region in regions) for Z in Zs]

results = Parallel(n_jobs=8)(delayed(reproducibility_across_regions)(all_labels, i, n_reg) for i, n_reg in enumerate(regions))

results = pd.DataFrame(results, columns=['ars', 'n'])
results['type'] = 'raw'

all_results = pd.concat([results_scaled, results])
all_results['algorithm'] = 'ward'

all_results.to_csv('/home/delavega/projects/clustering/results/bootstrap/hierarchical/ns_ward_reliability.csv')
