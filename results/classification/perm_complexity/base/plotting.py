#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


nine_colors = [(0.89411765336990356, 0.10196078568696976, 0.10980392247438431),
 (0.65845446095747107, 0.34122261685483596, 0.1707958535236471),
   (1.0, 0.50591311045721465, 0.0031372549487095253),
 (0.21602460800432691, 0.49487120380588606, 0.71987698697576341),
 (0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
               (0.400002384185791, 0.4000002384185791, 0.40000002384185791), 

 (0.60083047361934883, 0.30814303335021526, 0.63169552298153153),
       
               (0.99850826852461868, 0.60846600392285513, 0.8492888871361229),
    (0.99315647868549117, 0.9870049982678657, 0.19915417450315812)
 
 ]

def density_plot(data, file_name=None, covariance_factor=.2):
    """ Generate a density plot """
    data = np.array(data)
    density = stats.gaussian_kde(data)
    xs = np.linspace(0, data.max() + data.max() / 10, 200)
    density.covariance_factor = lambda: covariance_factor
    density._compute_covariance()
    plt.plot(xs, density(xs))

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

def plot_polar(data, n_top=3, selection='top', overplot=False, labels=None,
               palette='husl', metric='euclidean', label_size=26, threshold=None, max_val=None):

    r = np.linspace(0, 10, num=100)
    n_panels = data.shape[1]

    if selection == 'top':
        labels = []
        for i in range(n_panels):
            labels.extend(data.iloc[:, i].order(ascending=False) \
                .index[:n_top])
        labels = np.unique(labels)
    elif selection == 'std':
        labels = data.T.std().order(ascending=False).index[:n_top]

    data = data.loc[labels,:]
    
    # Use hierarchical clustering to order
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list
    dists = pdist(data, metric=metric)
    pairs = linkage(dists)
    pairs[pairs < 0] = 0
    order = leaves_list(pairs)
    data = data.iloc[order,:]
    labels = [labels[i] for i in order]


    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    
    ## Add first
    theta = np.concatenate([theta, [theta[0]]])
    if overplot:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        fig.set_size_inches(10, 10)
    else:
        fig, axes = plt.subplots(n_panels, 1, sharex=False, sharey=False,
                             subplot_kw=dict(polar=True))
        fig.set_size_inches((6, 6 * n_panels))

        
    from seaborn import color_palette
    colors = color_palette(palette, n_panels)
    for i in range(n_panels):
        if overplot:
            alpha = 0.05
        else:
            ax = axes[i]
            alpha = 0.8

        if max_val is None:
            max_val = data.values.max()
        
        ax.set_ylim(data.values.min(), max_val)
        
        d = data.iloc[:,i].values
        d = np.concatenate([d, [d[0]]])
        
        ax.plot(theta, d, alpha=1, color='black', linewidth=5)
        ax.plot(theta, d, alpha=0.9, color=colors[i], linewidth=4)
        
        ax.fill(theta, d, ec='k', alpha=alpha, color=colors[i], linewidth=0)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=label_size)
        [lab.set_fontsize(18) for lab in ax.get_yticklabels()]

    
    if threshold is not None:
        theta = np.linspace(0.0, 2 * np.pi, 999, endpoint=False)
        theta = np.concatenate([theta, [theta[0]]])
        d = np.array([threshold] * 1000)
        ax.plot(theta, d, alpha=1, color='black', linewidth=2, linestyle='--')


    plt.tight_layout()


def heat_map(data, x_labels, y_labels, size = (11, 11), file_name=None, lines=None, lines_off = 0, cmap='YlOrRd'):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.get_cmap(cmap), alpha=0.8)
    # from IPython.core.debugger import Tracer
    # Tracer()()

    fig = plt.gcf()

    fig.set_size_inches(size)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_yticklabels(y_labels, minor=False)
    ax.set_xticklabels(x_labels, minor=False)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    if lines is not None:
        xl, xh=ax.get_xlim()
        yl, yh=ax.get_ylim()
        if lines_off is not None:
            yl -= lines_off
            xh -= lines_off
        ax.hlines(lines, xl, xh, color='w', linewidth = 1.5)
        ax.vlines(lines, yl, yh, color='w', linewidth = 1.5)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    if file_name is None:
        fig.show()
    else:
        fig.savefig(file_name)


def save_region_importance_plots(clf, basename, thresh=20):
    for i in range(1, clf.mask_num):
        clf.plot_importances(
            i - 1, file_name=basename + "_imps_" + str(i) + ".png", thresh=thresh)
        clf.plot_importances(
            None, file_name=basename + "_imps_overall.png", thresh=thresh)

def plot_roc(y_test, y_pred):
    import pylab as pl
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


# def plot_min_max_fi(clf):
#     density_plot(
#         clf.feature_importances[tuple(np.where(sh_1 == sh_1.min())[0])],
#         file_name="../results/diagonstic/sh_1_min.png")
#     density_plot(
#         clf.feature_importances[tuple(np.where(sh_1 == sh_1.max())[0])],
#         file_name="../results/diagonstic/sh_1_max.png")
#     density_plot(clf.feature_importances[np.where(sh == sh.max())[0]][
#                  :, np.where(sh == sh.max())[1]], file_name="../results/diagonstic/sh_0_max.png")
#     density_plot(clf.feature_importances[np.where(sh == sh.min())[0]][
#                  :, np.where(sh == sh.min())[1]], file_name="../results/diagonstic/sh_0_min.png")
