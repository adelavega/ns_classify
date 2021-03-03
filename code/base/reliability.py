from sklearn.metrics import adjusted_rand_score

def reproducibility_rating(labels):
    """ Run mutliple pairwise supervised ratings to obtain an average
    rating
    Parameters
    ----------
    labels: list of label vectors to be compared
    Returns
    -------
    av_score:  float, a scalar summarizing the reproducibility of pairs of maps
    """
    av_score = 0
    niter = len(labels) 
    for i in range(1, niter):
        for j in range(i):
            av_score += adjusted_rand_score(labels[j], labels[i])
            av_score += adjusted_rand_score(labels[i], labels[j])
    av_score /= (niter * (niter - 1))

def reproducibility_across_regions(all_labels, i, n_reg):
    return [reproducibility_rating(zip(*all_labels)[i]), n_reg]