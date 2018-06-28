import sys
sys.path.append('../../../optimaltransport')

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.decomposition import PLDA

"""
Test that the PLDA class can be used with scikit-learn's GridSearchCV.

Liam Cattell -- June 2018
"""

# Load and reshape sample data
X, y = adni.load_data()
n_imgs, h, w = X.shape
X = X.reshape((n_imgs, h*w))

# Perform PCA before classification
pca = PCA()
X_pca = pca.fit_transform(X)

# Gridsearch of PLDA alpha parameter
params = {'alpha': [0.01, 0.1, 1]}
clf = GridSearchCV(PLDA(), params, verbose=2)
clf.fit(X_pca, y)

print('BEST SCORE: ', clf.best_score_)
print('BEST EST:', clf.best_estimator_)
