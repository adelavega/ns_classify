#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import datetime
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from base.tools import Logger
from base.pipelines import pipeline
from base.classifiers import PairwiseClassifier, OnevsallClassifier

from neurosynth.base.dataset import Dataset
from sklearn.metrics import roc_auc_score

now = datetime.datetime.now()

n_topics = 60
dataset = Dataset.load('../data/0.6/datasets/db_v6_topics-%d.pkl' % n_topics)
# cognitive_topics = ['topic' + topic[0] for topic in csv.reader(
# 	open('../data/unprocessed/abstract_topics_filtered/topic_sets/topic_keys'  + str(topics) + '-july_cognitive.csv', 'rU')) if topic[1] == "T"]

# junk_topics = ['topic' + topic[0] for topic in csv.reader(
# 	open('../data/unprocessed/abstract_topics_filtered/topic_sets/topic_keys' + str(topics) + '-july_cognitive.csv', 'rU')) if topic[1] == "F"]

# Analyses
def complete_analysis(dataset, dataset_name, name, masklist, processes = 1, features=None):
	for t in [0.05]:
		pipeline(
			OnevsallClassifier(dataset, masklist, thresh = t, thresh_low = 0, classifier=GaussianNB(), memsave=True),
			"classification/" + name + "_GNB_t" +str(t) + "_" + dataset_name, 
			features=features, processes=processes, scoring = roc_auc_score)

for i in [70]:
	complete_analysis(dataset, 
		"all_topics_%d" % n_topics, "LFC_scaled_%d" % i, 
		"../results/clustering/hierarchical/fastward_v6_scaled_LFC/l_%d.nii.gz" %i, 
		processes = 1)
