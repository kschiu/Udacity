#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
def author_id_accuracy(features_train, features_test, labels_train, labels_test):
		clf = GaussianNB()

		t0 = time()
		# Fit classifier on training data.
		clf.fit(features_train, labels_train)
		print "training time:", round(time()-t0, 3), "s"
		# ~2.5 Sec

		t0 = time()
		pred = clf.predict(features_test)
		print pred
		print "predicting time:", round(time()-t0, 3), "s"
		# ~.2 Sec

		# Return accuracy of trained data
		return clf.score(features_test, labels_test)

print author_id_accuracy(features_train, features_test, labels_train, labels_test)


#########################################################


