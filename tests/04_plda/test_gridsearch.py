import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.decomposition import PLDA


X, y = adni.load_data()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs, h*w))

pca = PCA()
X_pca = pca.fit_transform(X)

params = {'alpha': [0.01, 0.1, 1]}
clf = GridSearchCV(PLDA(), params, verbose=2)

clf.fit(X_pca, y)

print('BEST SCORE: ', clf.best_score_)
print('BEST EST:', clf.best_estimator_)
