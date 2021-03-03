import numpy as np
from tools import ProgressBar
from joblib import Parallel, delayed
import pandas as pd

def permutation_parallel(X, y, cla, feat_names, region, i):   
    newY = np.random.permutation(y)
    cla_fits = cla.fit(X, newY)
    fit_w = np.log(cla_fits.theta_[1] / cla_fits.theta_[0])
    
    results = []
    for n, lo in enumerate(fit_w):
        results.append([region + 1, i, feat_names[n], lo])
        
    return results

def permute_log_odds(clf, boot_n, feature_names=None, n_jobs=1):
    def z_score_array(arr, dist):
        return np.array([(v - dist[dist.region == i + 1].lor.mean()) / dist[dist.region == i + 1].lor.std() 
                         for i, v in enumerate(arr.tolist())])
                                           
    pb = ProgressBar(len(clf.data), start=True)
    overall_results = []
    
    if feature_names is None:
        feature_names = clf.feature_names

    for reg, (X, y) in enumerate(clf.data):
        results = Parallel(n_jobs = n_jobs)(delayed(permutation_parallel)(
            X, y, clf.classifier, feature_names, reg, i) for i in range(boot_n))
        for result in results:
            for res in result:
                overall_results.append(res)
        pb.next()
                                               
    perm_results = pd.DataFrame(overall_results, columns=['region', 'perm_n', 'topic_name', 'lor'])
    lor = pd.DataFrame(clf.odds_ratio, index=range(1, clf.odds_ratio.shape[0] + 1), columns=feature_names)
                                           
    
    return lor.apply(lambda x: z_score_array(x, perm_results[perm_results.topic_name == x.name]))

def bootstrap_parallel(X, y, cla, feat_names, region, i):
    ## Split into classes
    X0 = X[y == 0]
    X1 = X[y == 1]

    ## Sample with replacement from each class
    X0_boot = X0[np.random.choice(X0.shape[0], X0.shape[0])]
    X1_boot = X1[np.random.choice(X1.shape[0], X1.shape[0])]

    # Recombine
    X_boot = np.vstack([X0_boot, X1_boot])
    
    cla_fits = cla.fit(X_boot, y)
    fit_w = np.log(cla_fits.theta_[1] / cla_fits.theta_[0])
    
    results = []
    for n, lo in enumerate(fit_w):
        results.append([region, i, feat_names[n], lo])
        
    return results
def bootstrap_log_odds(clf, boot_n, feature_names=None, region_names = None, n_jobs=1):
    from statistics import percentile

    pb = ProgressBar(len(clf.data), start=True)

    if feature_names is None:
        feature_names = clf.feature_names

    if region_names is None:
        region_names = range(1, len(clf.data))

    overall_boot = []
    for reg, (X, y) in enumerate(clf.data):
        results = Parallel(n_jobs = n_jobs)(delayed(bootstrap_parallel)(
            X, y, clf.classifier, feature_names, region_names[reg], i) for i in range(boot_n))
        for result in results:
            for res in result:
                overall_boot.append(res)
        pb.next()
            
    overall_boot = pd.DataFrame(overall_boot, columns=['region', 'perm_n', 'topic_name', 'fi'])

    return overall_boot.groupby(['region', 'topic_name'])['fi'].agg({'mean' : np.mean, 'low_ci' : percentile(2.5), 'hi_ci' : percentile(97.5)}).reset_index()