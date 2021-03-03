#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


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


