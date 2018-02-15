import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from optrans.decomposition import PLDA

def plot_transform(X, y):
    ax = plt.subplot(111)
    for lab,mk,col in zip(range(4),('^', 's', 'o'),('blue', 'red', 'green')):
        plt.scatter(x=X[y==lab,0], y=X[y==lab,1], marker=mk, color=col)

    plt.xlabel('Dir. 1')
    plt.ylabel('Dir. 2')

    plt.grid()
    plt.tight_layout
    plt.show()
    return

X, y = load_iris(return_X_y=True)

plda = PLDA(alpha=10., n_components=2)
X_plda = plda.fit_transform(X, y)
plot_transform(X_plda, y)
print(plda.components_)
print("PLDA acc: {:.3f}".format(plda.score(X, y)))

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
plot_transform(X_lda, y)
prob_lda = lda.predict_log_proba(X)
print("LDA acc: {:.3f}".format(lda.score(X, y)))

# pca = PCA()
# X_pca = pca.fit_transform(X)
# plot_transform(X_pca, y)
