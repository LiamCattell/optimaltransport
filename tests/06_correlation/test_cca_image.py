import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split

from optrans.datasets import oasis

X, y, metadata = oasis.load_data()

ind = ~np.isnan(metadata).any(axis=1)
X = X[ind]
y = y[ind]
m = metadata[ind,1]

n_samples, h, w = X.shape
X = X.reshape((n_samples,h*w))

Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, m, test_size=0.5,
                                                random_state=43)

pca = PCA(n_components=10)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

cca = CCA(n_components=1)
cca.fit(Xtr_pca, mtr.reshape(-1,1))
Xtr_cca = np.squeeze(cca.transform(Xtr_pca))
Xte_cca = np.squeeze(cca.transform(Xte_pca))

# Correlation coefficient of training and test data
score_tr = np.corrcoef(Xtr_cca, mtr)[0,1]
score_te = np.corrcoef(Xte_cca, mte)[0,1]

# Get x-y coordinates of correlation line
coef = np.polyfit(Xte_cca, mte, 1)
line = coef[0] * Xte_cca + coef[1]
imin = Xte_cca.argmin()
imax = Xte_cca.argmax()
xx = [Xte_cca[imin], Xte_cca[imax]]
yy = [line[imin], line[imax]]


# Plot data projected on to first canonical direction
plt.scatter(Xtr_cca, mtr, c='b', label='train')
plt.scatter(Xte_cca, mte, c='r', label='test')
plt.plot(xx, yy, 'k--', lw=2, label='corr. test')
plt.xlabel('X scores')
plt.ylabel('m scores')
plt.title('R_tr = {:.2f}\nR_te = {:.2f}'.format(score_tr, score_te))
plt.legend()
plt.show()
