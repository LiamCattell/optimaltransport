import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from optrans.decomposition import PLDA

"""
Compare linear discriminant analysis (LDA), penalized LDA (PLDA), and principal
component analysis (PCA) using Fisher's iris dataset.

Liam Cattell -- March 2018
"""

def plot_transform(X, y, ax):
    # Plot transformed data as a scatter plot
    ax.grid(True, alpha=0.3)
    for lab in np.unique(y):
        ax.scatter(X[y==lab,0], X[y==lab,1])
    return

# Load iris dataset and labels
X, y = load_iris(return_X_y=True)

# Initialize the figure
fig, ax = plt.subplots(2, 3)

# Compute PLDA using different alpha values
for i,alpha in enumerate([0.,10.,1000.]):
    # Fit and transform data using PLDA
    plda = PLDA(alpha=alpha, n_components=2)
    X_plda = plda.fit_transform(X, y)

    # Compute classification accuracy
    acc = plda.score(X, y)

    # Plot transformed data
    plot_transform(X_plda, y, ax[0,i])
    ax[0,i].set_title("PLDA $\\alpha$={:.1f}\nacc={:.3f}".format(alpha, acc))

# For comparison, perform LDA
# Note: This should be the same as PLDA with alpha=0
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
acc = lda.score(X, y)
plot_transform(X_lda, y, ax[1,0])
ax[1,0].set_title("LDA\nacc={:.3f}".format(acc))

# For comparison, perform PCA
# Note: This should be the same as PLDA with very large alpha
pca = PCA()
X_pca = pca.fit_transform(X)
plot_transform(X_pca, y, ax[1,-1])
ax[1,2].set_title("PCA\nacc=N/A")

# Ignore the middle subplot
ax[1,1].axis('off')

plt.tight_layout()
plt.show()
