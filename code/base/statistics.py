#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
    
def specificity_score(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0] * 1.0 / confusion_matrix(y_true, y_pred)[:, 0].sum()

def roc_report(y_true, y_pred):
    return{'roc_auc': roc_auc_score(y_true, y_pred),
       'sensitivity' : recall_score(y_true, y_pred),
       'specificity' : specificity_score(y_true, y_pred)}
    
def mean_roc_report(folds):
    folds = [(a, b.squeeze()) for a, b in folds]
    return{'roc_auc': np.array([roc_auc_score(*fold) for fold in folds]).mean(),
           'sensitivity' : np.array([recall_score(*fold) for fold in folds]).mean(),
           'specificity' : np.array([specificity_score(*fold) for fold in folds]).mean()}

def shannons(data):
    """ Given a list of frequencies, returns the SDI"""
    
    from math import log as ln

    # from IPython.core.debugger import Tracer
    # Tracer()()
    
    def p(n, N):
        """ Relative abundance """
        if n is  0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)
            
    N = sum(data)
    
    return -sum(p(n, N) for n in data if not n == 0)


def get_roc(x, y):
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, thresholds = roc_curve(x, y)
	return auc(fpr, tpr)

def dist(x, y):
  #1d only
  return np.abs(x[:, None] - y)

def d_n(x):
  d = dist(x, x)
  dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean() 
  return dn

def dcov_all(x, y):
    # Coerce type to numpy array if not already of that type.
    try: x.shape
    except AttributeError: x = np.array(x)
    try: y.shape
    except AttributeError: y = np.array(y)

    dnx = d_n(x)
    dny = d_n(y)

    denom = np.product(dnx.shape)
    dc = np.sqrt((dnx * dny).sum() / denom)
    dvx = np.sqrt((dnx**2).sum() / denom)
    dvy = np.sqrt((dny**2).sum() / denom)
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy

def dcor(x,y):
    return dcov_all(x,y)[1]



