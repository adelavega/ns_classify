from base.mv import best_subsets
from base.classifiers import OnevsallContinuous
from sklearn.metrics import roc_auc_score

o_clf = OnevsallContinuous.load('classifier.pkl')

best_subsets(o_clf, roc_auc_score, outfile='results/nine_regions.csv')
