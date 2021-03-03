import itertools
import tools
import pandas as pd
from sklearn.cross_validation import cross_val_score
import numpy as np
from neurosynth.analysis import classify


def calc_mv_parallel(args):
    ((X, ys), classifier, scorer, fis, method), reg = args

    y = ys[reg]
    fis = fis[reg]

    n_topics = X.shape[1]

    if method == 'sequential':
        results = []
        for i in range(1, n_topics):
            X_1 = X[:, np.abs(fis).argsort()[::-1][0:i]]
            results.append(
                [cross_val_score(classifier, X_1, y, cv=4, scoring=scorer).mean(), i, reg])
    elif method == 'combinatorial':
        results = []
        ix = [np.abs(fis).argsort()[::-1][0]]
        remaining = range(1, n_topics+1)
        remaining.remove(ix[0])

        for i in range(1, n_topics):
            if i == 1:
                X_1 = X[:, ix]
                results.append(
                    [cross_val_score(classifier, X_1, y, cv=4, scoring=scorer).mean(), i, reg])
            else:
                test_results = []
                for num, new_feat in enumerate(remaining):
                    try_comb = ix + [new_feat]
                    X_1 = X[:, try_comb]
                    test_results.append(
                        [cross_val_score(classifier, X_1, y, cv=4, scoring=scorer).mean(), i, reg])

                test_results = np.array(test_results)

                winner = test_results[:, 0] == test_results[:, 0].max()

                results.append(test_results[winner].tolist()[0])

                remaining.remove(np.where(winner)[0][0] + 1)

    return results


def calc_mv(clf, scorer, regions=None, processes=7, method='sequential'):
    n_regions = clf.data[1].shape[0]
    if regions is None:
        regions = range(0, n_regions)

    if processes > 1:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        pool = itertools

    pb = tools.ProgressBar(len(regions), start=True)

    overall_results = []
    for result in pool.imap(calc_mv_parallel, itertools.izip(itertools.repeat((clf.data, clf.classifier, scorer, clf.feature_importances, method)), regions)):
        pb.next()
        for row in result:
            overall_results.append(row)

    return pd.DataFrame(overall_results, columns=['score', 'num_features', 'region'])


def calc_mv_parallel_classifier(args):
    (filename, classifier, scorer, comp_dims,
     fis, feature_names, method), reg = args

    X, y = np.memmap(filename, dtype='object', mode='r',
                     shape=comp_dims)[reg]
    fis = fis[reg]

    n_topics = X.shape[1]

    if method == 'sequential':
        results = []
        ix = np.abs(fis).argsort()[::-1]
        for i in range(1, n_topics):
            X_1 = X[:, np.abs(fis).argsort()[::-1][0:i]]
            feature = feature_names[ix[i-1]]
            output = classify.classify(
                X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
            results.append([output['score'], i, reg, feature])

    elif method == 'combinatorial':
        results = []
        ix = [np.abs(fis).argsort()[::-1][0]]
        remaining = range(0, n_topics)
        remaining.remove(ix[0])

        for i in range(1, n_topics + 1):
            if i == 1:
                X_1 = X[:, ix]
                feature = feature_names[ix[i-1]]
                output = classify.classify(
                    X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
                results.append([output['score'], i, reg, feature])
            else:
                test_results = []
                features = []
                for num, new_feat in enumerate(remaining):
                    try_comb = ix + [new_feat]
                    X_1 = X[:, try_comb]
                    feature = feature_names[new_feat]
                    output = classify.classify(
                        X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
                    test_results.append([output['score'], i, reg, feature])
                    features.append(new_feat)

                test_results = pd.DataFrame(test_results)

                winner = test_results.ix[:, 0] == test_results.ix[:, 0].max()

                results.append(map(list, test_results[winner].values)[0])

                remaining.remove(features[np.where(winner)[0][0]])

                ix += [features[np.where(winner)[0][0]]]

    return results


def calc_mv_classifier(clf, scorer, regions=None, processes=7, method='sequential'):
    import os.path as path
    from tempfile import mkdtemp

    n_regions = clf.data.shape[0]
    if regions is None:
        regions = range(0, n_regions)

    if processes != 1:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        pool = itertools

    pb = tools.ProgressBar(len(regions), start=True)

    filename = path.join(mkdtemp(), 'data.dat')
    data = np.memmap(filename, dtype='object', mode='w+', shape=clf.comp_dims)
    data[:] = clf.data[:]

    overall_results = []
    for result in pool.imap(calc_mv_parallel_classifier, itertools.izip(itertools.repeat((filename, clf.classifier, scorer,
                                                                                          clf.comp_dims, clf.feature_importances, np.array(clf.feature_names), method)), regions)):
        pb.next()
        for row in result:
            overall_results.append(row)

    overall_results = pd.DataFrame(
        overall_results, columns=['score', 'num_features', 'region', 'feature'])
    overall_results.region += 1
    return overall_results


def bootstrap_mv_parallel(args):

    try:
        (X, y, classifier, scorer, fis,
         feature_names, method, reg), boot_n = args

        n_topics = X.shape[1]

        np.random.seed()
        ran_index = np.random.choice(X.shape[0], X.shape[0])

        # Bootstrap sample X & y
        X = X[ran_index, :]
        y = y[ran_index]

        if method == 'sequential':
            results = []
            ix = np.abs(fis).argsort()[::-1]
            for i in range(1, n_topics):
                X_1 = X[:, np.abs(fis).argsort()[::-1][0:i]]
                feature = feature_names[ix[i-1]]
                output = classify.classify(
                    X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
                results.append([output['score'], i, reg, feature])

        elif method == 'combinatorial':
            results = []
            ix = []  # Feature order index
            remaining = range(0, n_topics)

            for i in range(0, n_topics):
                test_results = []
                for num, new_feat in enumerate(remaining):
                    try_comb = ix + [new_feat]
                    X_1 = X[:, try_comb]
                    feature = feature_names[new_feat]
                    output = classify.classify(
                        X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
                    test_results.append(
                        [output['score'], i, feature, reg, boot_n, new_feat])

                test_results = pd.DataFrame(test_results)

                winner = test_results[
                    test_results.ix[:, 0] == test_results.ix[:, 0].max()]

                if winner.shape[0] > 1:
                    winner = winner.iloc[0]

                results.append(map(list, winner.values)[0][0:5])

                remaining.remove(winner[5].values)

                ix += winner[5].values.tolist()

        elif method == 'shannons':
            from base.statistics import shannons
            clf = classify.classify(
                X, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='clf')
            odds_ratios = np.log(clf.clf.theta_[1] / clf.clf.theta_[0])
            odds_ratios -= (odds_ratios.min() - 0.000001)
            results = [shannons(odds_ratios), reg, boot_n]

    except:
        import warnings
        warnings.warn('something went wrong')

        results = None
    finally:
        return results


def bootstrap_mv(clf, scorer, regions=None, processes=None, boot_n=100, method='sequential', outfile=None):
    """ This function calculates "complexity curves" for each region in clf. The complexity curves are done on a 
    boostrapped sample boot_n times """

    n_regions = clf.data.shape[0]
    if regions is None:
        regions = range(0, n_regions)

    if processes != 1:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        pool = itertools

    pb = tools.ProgressBar(len(regions) * boot_n, start=True)

    if method == 'shannons':
        cols = ['shannons', 'region', 'boot_n']
    else:
        cols = ['score', 'num_features', 'feature', 'region', 'boot_n']

    overall_results = []
    # Do this for every region sequentially
    for i, (X, y) in enumerate(clf.data[regions]):
        for result in pool.imap(bootstrap_mv_parallel, itertools.izip(itertools.repeat((
                X, y, clf.classifier, scorer, clf.feature_importances, np.array(clf.feature_names), method, i)), range(boot_n))):
            pb.next()

            if result is not None:
                if method != 'shannons':
                    for row in result:
                        overall_results.append(row)
                else:
                    overall_results.append(result)

                if outfile is not None:
                    pd.DataFrame(overall_results, columns=cols).to_csv(outfile)

    overall_results = pd.DataFrame(overall_results, columns=cols)
    overall_results.region += 1
    return overall_results


def bootstrap_mv_full_parallel(args):

    try:
        (X, y_high, y_low, classifier, scorer, method), boot_n = args

        np.random.seed()
        ran_index = np.random.choice(X.shape[0], X.shape[0])

        from neurosynth.analysis.classify import regularize

        # Bootstrap sample X & y
        X = X.iloc[ran_index, :]
        y_high = pd.DataFrame(y_high[:, ran_index])
        y_low = pd.DataFrame(y_low[:, ran_index])

        feature_names = X.columns.tolist()
        n_topics = len(feature_names)

        X = regularize(X, method='scale')
        results = []
        for reg_i, reg_y_high in y_high.iterrows():
            reg_ix = (
                (y_low.iloc[reg_i, :] == True) & (reg_y_high == False)) == False
            reg_y = reg_y_high[reg_ix].astype('int')
            reg_X = X[reg_ix.values, :]

            if method == 'combinatorial':

                ix = []  # Feature order index
                remaining = range(0, n_topics)

                for i in range(0, n_topics):
                    test_results = []
                    for num, new_feat in enumerate(remaining):
                        try_comb = ix + [new_feat]
                        X_1 = reg_X[:, try_comb]
                        feature = feature_names[new_feat]
                        output = classify.classify(
                            X_1, reg_y.values, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')
                        test_results.append(
                            [output['score'], i, feature, reg_i, boot_n, new_feat])

                    test_results = pd.DataFrame(test_results)

                    winner = test_results[
                        test_results.ix[:, 0] == test_results.ix[:, 0].max()]

                    if winner.shape[0] > 1:
                        winner = winner.iloc[0]

                    results.append(map(list, winner.values)[0][0:5])

                    remaining.remove(winner[5].values)

                    ix += winner[5].values.tolist()

            # elif method == 'shannons':
            #     from base.statistics import shannons
            #     clf = classify.classify(X, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='clf')
            #     odds_ratios = np.log(clf.clf.theta_[1] / clf.clf.theta_[0])
            #     odds_ratios -= (odds_ratios.min() - 0.000001)
            #     results = [shannons(odds_ratios), reg, boot_n]
    except:
        import warnings
        warnings.warn('something went wrong')

        results = None
    finally:
        return results


def bootstrap_mv_full(dataset, clf, scorer, mask, features=None, processes=None, boot_n=100, method='combinatorial', outfile=None, thresh_high=0.05, thresh_low=0):
    from neurosynth.analysis.reduce import average_within_regions

    if processes != 1:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        pool = itertools

    pb = tools.ProgressBar(boot_n, start=True)

    if method == 'shannons':
        cols = ['shannons', 'region', 'boot_n']
    else:
        cols = ['score', 'num_features', 'feature', 'region', 'boot_n']

    X = dataset.get_feature_data(features=features)
    y_high = average_within_regions(dataset, mask, threshold=thresh_high)
    y_low = average_within_regions(dataset, mask, threshold=thresh_low)
    # Get feature names
    overall_results = []
    for result in pool.imap(bootstrap_mv_full_parallel, itertools.izip(itertools.repeat((
            X, y_high, y_low, clf, scorer, method)), range(boot_n))):
        pb.next()

        if result is not None:
            if method != 'shannons':
                for row in result:
                    overall_results.append(row)
            else:
                overall_results.append(result)

            if outfile is not None:
                pd.DataFrame(overall_results, columns=cols).to_csv(outfile)
                print "Saved"

    overall_results = pd.DataFrame(overall_results, columns=cols)
    overall_results.region += 1
    return overall_results


def minimum_percent(region, percent=.1):
    max = region['score'].max()
    labeled_rows = (region['score'] > (max - ((max) * percent)))
    return region[labeled_rows]['num_features'].min()


def minimum_fixed(region, fixed=.01):
    labeled_rows = (region['score'] > (region['score'].max() - fixed))
    return region[labeled_rows]['num_features'].min()

from sklearn.naive_bayes import GaussianNB


class NaivesBayesWrapper(GaussianNB):

    def __init__(self, metric='log-odds'):
        self.metric = metric
        super(NaivesBayesWrapper, self).__init__()

    def fit(self, X, y):
        clf = super(NaivesBayesWrapper, self).fit(X, y)

        if self.metric == 'log-odds':
            self.coef_ = np.log(self.theta_[1] / self.theta_[0])
        elif self.metric == 'difference':
            self.coef_ = self.theta_[1] - self.theta_[0]

        return clf


def best_subsets_parallel(args):
    (X, y, classifier, scorer, feature_names), comb = args

    X_1 = X[:, comb]

    features = list(np.array(feature_names)[list(comb)])
    output = classify.classify(
        X_1, y, classifier=classifier, cross_val='4-Fold', scoring=scorer, output='summary')

    return (output['score'], features, comb)


def best_subsets(clf, scorer, processes=None, outfile=None):
    if processes == None:
        from multiprocessing import Pool
        pool = Pool()
        print pool._processes
    elif processes > 1:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        pool = itertools

    classifier = NaivesBayesWrapper()

    n_topics = len(clf.feature_names)

    results = []

    for n_reg, (X, y) in enumerate(clf.data):
        for n_comb in range(1, n_topics):
            print "Region: " + str(n_reg) + " NComb: " + str(n_comb)
            combinations = list(
                itertools.combinations(range(0, X.shape[1]), n_comb))

            test_results = []
            for result in pool.imap(best_subsets_parallel, itertools.izip(itertools.repeat((X, y, classifier, scorer, clf.feature_names)), combinations)):
                test_results.append(result)

            test_results = pd.DataFrame(test_results)

            winner = test_results.ix[:, 0] == test_results.ix[:, 0].max()

            # Add combination
            results.append(
                test_results[winner].values.tolist()[0] + [n_comb, n_reg])
            print winner

            pd.DataFrame(results, columns=[
                         'score', 'topics', 'combination', 'num_features', 'region']).to_csv(outfile)

    results = pd.DataFrame(
        results, columns=['score', 'topics', 'combination', 'num_features', 'region'])

    results.region += 1
    return results
