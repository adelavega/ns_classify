"""
This is a fork from parietal.probabilistic_parcellation.mixed_effects_parcel
Do not use it if you ahve access to the original one.

This module can be used to fit a mixed effects model on parcellations.
This will be used for statistical inference and model selection
in parcellations.

The basis consists in recovering the true parameters of a mixed effects models
to model the signal given a certain parcellation.
This is typically detailed oin the example below:

>>> from parietal.probabilistic_parcellation.mixed_effects_parcel import *
>>> n_subjects, n_voxels = 10, 10
>>> mu, sigma1, sigma2 = 3., 1., 1.
>>> y, u = generate_data(n_subjects=n_subjects, n_voxels=n_voxels, mu=mu,
>>>                       sigma1=sigma1, sigma2=sigma2)
>>> print 'simulated: ', log_likelihood_(y, mu, sigma1, sigma2, u)
>>> mu, sigma1, sigma2 =  em_inference(y, u)
>>> print 'em: ', log_likelihood_(y, mu, sigma1, sigma2, u), mu, sigma1, sigma2
>>> mu, sigma1, sigma2, ll = manual_optimization(y, u)
>>> print 'grid search: ', ll, mu, sigma1, sigma2

This is used to validate and choose the optimal spatial model on a given
dataset.

Author: Bertrand Thirion, 2012
Edited 2016 by Alejandro de la Vega to use w/ Neurosynth
"""

import numpy as np
from scipy.stats import norm
from sklearn import metrics


def score_spatial_model(X, label, cv=None, two_level=False, null=False):
    """Give a score to a data labelling With/out cross-validation

    Parameters
    ==========
    X: array of shape(n_voxels, n_subjects) the data to be parcelled
    label: array of shape (n_voxels) an index array describing the parcellation
    cv: string, optional,
         cross validation scheme, one of (None, 'loo', 'kfold', 'll', 'log_lr')
    two_level: bool, optional,
               whether a one-or two level variance partition scheme is used
    null: bool, optional
          whether the likelihood is estimated under H0 (mu=0) or not

    Returns
    =======
    score: float, the sumed log-likelihood of the data under the parcellation
    """
    from sklearn.cross_validation import LeaveOneOut, KFold
    score = 0

    if cv in ['bic', 'll', None]:
        ll, _, _, _, bic = parameter_map(X, label, two_level, null)
        if cv == 'bic':
            score = bic.sum()
        else:
            score = ll.sum()
    elif cv == 'log_lr':
        ll1, _, _, _, _ = parameter_map(X, label, two_level, False)
        ll2, _, _, _, _ = parameter_map(X, label, two_level, True)
        score = ll1.sum() - ll2.sum()
    elif cv in ['loo', 'kfold']:
        score = 0
        if cv == 'loo':
            cv = LeaveOneOut(X.shape[1])
        elif cv == 'kfold':
            cv = KFold(X.shape[1], min(10, X.shape[1]))
        for k in np.unique(label):
            for (train, test) in cv:
                mu = None
                if null:
                    mu = 0
                mu, sigma1, sigma2, _ = em_inference_regular(
                    X[label == k][:, train], two_level=two_level, mu=mu)
                test_ll = log_likelihood_regular(
                    X[label == k][:, test], mu, sigma1, sigma2,
                    two_level=two_level)
                score += test_ll
    else:
        raise ValueError('unknown keyword from evaluation scheme (cv argument)')
    return score


def parameter_map(X, label, two_level=True, null=False):
    """Return the likelihood of the model per label

    Parameters
    ==========
    X: array of shape(n_voxels, n_subjects) the data to be parcelled
    label: array of shape (n_voxels) an index array describing the parcellation
    two_level: bool, optional
               Whether the model contains two levels or variance or not.
    null: bool, optional,
          whether parameters should be estimated under the null (mu=0) or not

    Returns
    =======
    ll: array_of shape(len(np.unique(label)))
    mu: array_of shape(len(np.unique(label)))
    sigma1: array_of shape(len(np.unique(label)))
    sigma2: array_of shape(len(np.unique(label)))
    bic: array_of shape(len(np.unique(label)))
    """
    n_labels = len(np.unique(label))
    ll, bic = np.zeros(n_labels), np.zeros(n_labels)
    mu = np.zeros(n_labels)
    sigma1 = np.zeros(n_labels)
    sigma2 = np.zeros(n_labels)
    for k in np.unique(label):
        if null:
            mu[k], sigma1[k], sigma2[k], ll[k] = em_inference_regular(
                X[label == k], two_level=two_level, mu=0)
        else:
            mu[k], sigma1[k], sigma2[k], ll[k] = em_inference_regular(
                X[label == k], two_level=two_level)

        if two_level and (null == False):
            bic[k] = -2 * ll[k] + 3 * np.log(X[label == k].size)
        elif two_level and null:
            bic[k] = -2 * ll[k] + 2 * np.log(X[label == k].size)
        elif (two_level == False) and (null == False):
            bic[k] = -2 * ll[k] + 2 * np.log(X[label == k].size)
        else:
            bic[k] = -2 * ll[k] + np.log(X[label == k].size)
    return ll, mu, sigma1, sigma2, bic

def em_inference_regular(X, mu=None, sigma1=None, sigma2=None, niter=30,
                         eps=1.e-3, verbose=False, two_level=True):
    """Idem em_inference, but the data is presented as an
    (n_samples, n_subjects) array"""
    y = X.T.ravel()
    if two_level == False:
        if mu == None:
            mu = np.mean(y)
        sigma2 = np.sqrt(np.sum((y - mu) ** 2) / y.size)
        return mu, 0, sigma2
        
    u = np.repeat(np.arange(X.shape[1]), X.shape[0])
    return em_inference_fast(y, u, mu, sigma1, sigma2, niter, eps,
                             verbose=verbose)

def log_likelihood_map(X, label, mu, sigma1, sigma2):
    """Return the likelihood of the model per label

    Parameters
    ==========
    X: array of shape(n_voxels, n_subjects) the data to be parcelled
    label: array of shape (n_voxels) an index array describing the parcellation
    mu: array_of shape(len(np.unique(label)))
    sigma1: array_of shape(len(np.unique(label)))
    sigma2: array_of shape(len(np.unique(label)))

    Returns
    =======
    ll: array_of shape(len(np.unique(label)))
    """
    n_labels = len(np.unique(label))
    ll = np.zeros(n_labels)
    for k in np.unique(label):
        ll[k] = log_likelihood_regular(
            X[label == k], mu[k], sigma1[k], sigma2[k])
    return ll


def log_likelihood(y, mu, sigma1, sigma2, J):
    """ Return the data loglikelihood
    slow method, but works for all covariance matrices

    Parameters
    ==========
    y: array of shape(n_samples), the input data
    mu: float, mean parameter
    sigma1: float, first level variance
    sigma2: float, seond level variance
    J: covariance model ('second level' model)

    Returns
    =======
    ll: float, the log-likelihood of the data under the proposed model
    """
    I = np.eye(y.size)
    log_det = np.log(np.linalg.det(J * sigma2 ** 2 + I * sigma1 ** 2))
    prec = np.linalg.pinv((sigma2 ** 2) * J + (sigma1 ** 2) * I)
    quad = np.dot(np.dot(y - mu, prec), y - mu)
    return - 0.5 * (log_det + quad + y.size * np.log(2 * np.pi))


def log_likelihood_regular(X, mu, sigma1, sigma2, two_level=True):
    """Idem log_likelihood, but the data is repsented as an
    (n_samples, n_subjects) array"""
    y = X.T.ravel()
    if X.shape[0] == 1:
        # this is to avoid numerical instabilities
        return np.log(norm.pdf(y, loc=mu, scale=sigma2)).sum()
    if two_level == False:
        return np.log(norm.pdf(y, mu, sigma2)).sum()
    u = np.repeat(np.arange(X.shape[1]), X.shape[0])
    return log_likelihood_(y, mu, sigma1, sigma2, u)


def log_likelihood_(y, mu, sigma1, sigma2, u):
    """ Return the data loglikelihood (fast method specific to the block model)

    Parameters
    ==========
    y: array of shape(n_samples), the input data
    mu: float, mean parameter
    sigma1: float, first level variance
    sigma2: float, seond level variance
    u: array of shape(n_samples), the associated index

    Returns
    =======
    ll: float, the log-likelihood of the data under the proposed model
    """
    log_det, quad = 0, 0
    for k in np.unique(u):
        pop = np.sum(u == k)
        log_det += np.log(pop * sigma2 ** 2 + sigma1 ** 2) +\
            (pop - 1) * np.log(sigma1 ** 2)
        prec = - 1. / (pop * sigma1 ** 2 +\
                            sigma1 ** 4 / sigma2 ** 2)
        quad += prec * (pop ** 2) * (np.mean(y[u == k]) - mu) ** 2
    quad += np.sum((y - mu) ** 2) / sigma1 ** 2
    return - 0.5 * (log_det + quad + y.size * np.log(2 * np.pi))


def log_likelihood_fast(s0, s1, s2, mu, sigma1, sigma2):
    """ Return the data loglikelihood (fast method specific to the block model)

    Parameters
    ==========
    s0, s1, s2: moments of order 0, 1 and 2 of the data
    mu: float, mean parameter
    sigma1: float, first level variance
    sigma2: float, seond level variance
    u: array of shape(n_samples), the associated index

    Returns
    =======
    ll: float, the log-likelihood of the data under the proposed model
    """
    log_det, quad = 0, 0
    for (s0_, s1_, s2_) in zip (s0, s1, s2):
        if s0_ > 0:
            log_det += np.log(s0_ * sigma2 ** 2 + sigma1 ** 2) +\
                (s0_ - 1) * np.log(sigma1 ** 2)
            prec = - 1. / (s0_ * sigma1 ** 2 + sigma1 ** 4 / sigma2 ** 2)
            quad += prec * (s0_ ** 2) * (s1_ / s0_ - mu) ** 2
            quad += (s2_ + mu * (s0_ * mu - 2 * s1_ )) / sigma1 ** 2
    return - 0.5 * (log_det + quad + s0.sum() * np.log(2 * np.pi))


def em_inference(y, u, mu=None, sigma1=None, sigma2=None, niter=30, eps=1.e-3,
                 mins=1.e-6, verbose=False):
    """use an EM algorithm to compute sigma1, sigma2, mu

    Parameters
    ==========
    y: array of shape(n_samples), the input data
    u: array of shape(n_samples), the associated index
    sigma1: float, optional, initial value for sigma1
    sigma2: float, optional, initial value for sigma2
    niter: int, optional, max number of EM iterations
    eps: float, optional, convergence criteria on log-likelihood
    mins: float, optional, lower bound on variance values (numerical issues)
    verbose: bool, optional, verbosity mode

    Returns
    =======
    mu: float, estimatred mean
    sigma1: float, estimated first-level variance
    sigma2: float, second-level variance
    """
    if mu is None:
        mu = y.mean()
        learn_mu = True
    else:
        learn_mu = False
    beta = np.array([np.mean(y[u == k]) for k in np.unique(u)])
    size = np.array([np.sum(u == k) for k in np.unique(u)])

    # initialization
    if sigma1 is None:
        sigma1 = np.sqrt(np.array([
                    np.sum((y[u == k] - beta[i]) ** 2)
                    for (i, k) in enumerate(np.unique(u))]).sum() / y.size)
    if sigma2 is None:
        sigma2 = np.std(beta)
    ll_old = - np.infty
    for j in range(niter):
        sigma1, sigma2 = np.maximum(sigma1, mins), np.maximum(sigma2, mins)
        ll = log_likelihood_(y, mu, sigma1, sigma2, u)
        if verbose:
            print j, log_likelihood_(y, mu, sigma1, sigma2, u)
        if ll < ll_old - eps and len(np.unique(u)) < u.size:
            raise ValueError('LL should not decrease during EM')
        if ll - ll_old < eps:
            break
        else:
            ll_old = ll
        var = 1. / (size * 1. / sigma1 ** 2 + 1. / sigma2 ** 2)
        cbeta = var * (size * beta / sigma1 ** 2 + mu / sigma2 ** 2)
        if learn_mu:
            mu = cbeta.mean()
        sigma2 = np.sqrt(((cbeta - mu) ** 2 + var).mean())
        sigma1 = np.sqrt(
            np.array([size[i] * var[i] + np.sum((y[u == k] - cbeta[i]) ** 2)
                      for (i, k) in enumerate(np.unique(u))]).sum() / y.size)
    return mu, sigma1, sigma2, ll


def em_inference_fast(y, u, mu=None, sigma1=None, sigma2=None, niter=30, 
                      eps=1.e-3, mins=1.e-6, verbose=False):
    """use an EM algorithm to compute sigma1, sigma2, mu -- fast version

    Parameters
    ==========
    y: array of shape(n_samples), the input data
    u: array of shape(n_samples), the associated index
    sigma1: float, optional, initial value for sigma1
    sigma2: float, optional, initial value for sigma2
    niter: int, optional, max number of EM iterations
    eps: float, optional, convergence criteria on log-likelihood
    mins: float, optional, lower bound on variance values (numerical issues)
    verbose: bool, optional, verbosity mode

    Returns
    =======
    mu: float, estimatred mean
    sigma1: float, estimated first-level variance
    sigma2: float, second-level variance
    ll: float, the log-likelihood of the data

    Note
    ====
    use summary statistics for speed-up
    """
    if mu is None:
        mu = y.mean()
        learn_mu = True
    else:
        learn_mu = False
    s0 = np.array([np.sum(u == k) for k in np.unique(u)])
    s1 = np.array([np.sum(y[u == k]) for k in np.unique(u)])
    s2 = np.array([np.sum(y[u == k] ** 2) for k in np.unique(u)])

    # initialization
    if sigma1 is None:
        sigma1 = np.sqrt((s2 - (s1 ** 2) / s0).sum() / y.size)
    if sigma2 is None:
        sigma2 = np.std(s1 / s0)
    
    # EM iterations
    ll_old = - np.infty
    for j in range(niter):
        sigma1, sigma2 = np.maximum(sigma1, mins), np.maximum(sigma2, mins)
        ll = log_likelihood_fast(s0, s1, s2, mu, sigma1, sigma2)
        #ll_ = log_likelihood_(y, mu, sigma1, sigma2, u)
        #assert(np.abs(ll - ll_) < 1.e-8)
        if verbose:
            print j, ll
        if ll < ll_old - eps and len(np.unique(u)) < u.size:
            raise ValueError('LL should not decrease during EM')
        if ll - ll_old < eps:
            break
        else:
            ll_old = ll
        var = 1. / (s0 * 1. / sigma1 ** 2 + 1. / sigma2 ** 2)
        cbeta = var * (s1 / sigma1 ** 2 + mu / sigma2 ** 2)
        if learn_mu:
            mu = cbeta.mean()
        sigma2 = np.sqrt(((cbeta - mu) ** 2 + var).mean())
        sigma1 = np.sqrt(
            (s0 * var + s2 +  cbeta * (s0 * cbeta - 2 * s1)).sum() / y.size)
    return mu, sigma1, sigma2, ll


def manual_optimization(y, u):
    """This function does a grid search to select the best parameters

    Parameters
    ==========
    y: array of shape(n_samples), the input data
    u: array of shape(n_samples), the associated index

    Returns
    =======
    mu: float, estimatred mean
    sigma1: float, estimated first-level variance
    sigma2: float, second-level variance
    maxll: float, maximum log-likelihood achieved
    """
    ub = np.log(np.maximum(np.var(y), 1.e-12)) / np.log(10)

    def _grid_search(u, y, sigma1_range, sigma2_range):
        ll = []
        for sigma1 in sigma1_range:
            for sigma2 in sigma2_range:
                ll.append(log_likelihood_(y, y.mean(), sigma1, sigma2, u))
        ll = np.array(ll).reshape(len(sigma2_range), len(sigma1_range))
        i, j = np.where(ll == ll.max())
        return sigma1_range[i], sigma2_range[j], ll[i, j]

    # coarse grid search
    sigma_range = np.logspace(ub - 2, ub, 10)
    sigma1, sigma2, maxll = _grid_search(u, y, sigma_range, sigma_range)

    # fine grid_search
    log1, log2 = np.log(sigma1) / np.log(10), np.log(sigma2) / np.log(10)
    sigma1_range = np.logspace(log1 - .2, log1 + .2, 10)
    sigma2_range = np.logspace(log2 - .2, log2 + .2, 10)
    sigma1, sigma2, maxll = _grid_search(u, y, sigma1_range, sigma2_range)
    return y.mean(), sigma1, sigma2, maxll




def supervised_rating(true_label, pred_label, score='ars', verbose=False):
    """Provide assessment of a certain labeling given a ground truth

    Parameters
    ----------
    true_label: array of shape (n_samples), the true labels
    pred_label: array of shape (n_samples), the predicted labels
    score: string, on of 'ars', 'ami', 'vm'
    """
    ars = metrics.adjusted_rand_score(true_label, pred_label)
    ami = metrics.adjusted_mutual_info_score(true_label, pred_label)
    vm = metrics.v_measure_score(true_label, pred_label)
    if verbose:
        print 'Adjusted rand score:', ars
        print 'Adjusted MI', ami
        print 'Homogeneity', metrics.homogeneity_score(true_label, pred_label)
        print 'Completeness', metrics.completeness_score(true_label, pred_label)
        print 'V-measure', vm
    if score == 'ars':
        return ars
    elif score == 'vm':
        return vm
    else:
        return ami


def reproducibility_rating(labels, score='ars', verbose=False):
    """ Run mutliple pairwise supervised ratings to obtain an average
    rating

    Parameters
    ----------
    labels: list of label vectors to be compared
    score: string, on of 'ars', 'ami', 'vm'
    verbose: bool, verbosity

    Returns
    -------
    av_score:  float, a scalar summarizing the reproducibility of pairs of maps
    """
    av_score = 0
    niter = len(labels) 
    for i in range(1, niter):
        for j in range(i):
            av_score += supervised_rating(labels[j], labels[i], score=score,
                                       verbose=verbose)
            av_score += supervised_rating(labels[i], labels[j], score=score,
                                       verbose=verbose)
    av_score /= (niter * (niter - 1))
    return av_score

