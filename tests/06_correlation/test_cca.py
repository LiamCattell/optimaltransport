import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA as CanonCorr
from sklearn.model_selection import train_test_split

from optrans.datasets import oasis
from optrans.decomposition import CCA

"""
Perform CCA on the OASIS brain MRI dataset metadata. Plots the data projected
on to the first canonical direction.
"""

# Get the OASIS metadata and discard any rows containing NaNs
_, _, metadata = oasis.load_data()
metadata = metadata[~np.isnan(metadata).any(axis=1)]

# Select the metadata to use
X = metadata[:,[1,4]]       # [age, mmse]
Y = metadata[:,6:9]       # [etiv, nwbv, asf]

# Split into training and testing data
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.25, random_state=42)

# Perform CCA and transform data
cca = CanonCorr(n_components=2)
Xtr_cca, Ytr_cca = cca.fit_transform(Xtr, Ytr)
Xte_cca, Yte_cca = cca.transform(Xte, Yte)

Ytmp = Ytr_cca[0].reshape(1,-1)
Ytr_recon = np.dot(Ytmp, cca.y_weights_.T) + cca.y_mean_
print('Y recon: ', Ytr_recon)
print('Y true:  ', Ytr[0])

# Get 1st direction for plotting
Xtr_cca0 = np.squeeze(Xtr_cca[:,0])
Ytr_cca0 = np.squeeze(Ytr_cca[:,0])
Xte_cca0 = np.squeeze(Xte_cca[:,0])
Yte_cca0 = np.squeeze(Yte_cca[:,0])

# Correlation coefficient of training and test data
score_tr = np.corrcoef(Xtr_cca0, Ytr_cca0)[0,1]
score_te = np.corrcoef(Xte_cca0, Yte_cca0)[0,1]

# Get x-y coordinates of correlation line
coef = np.polyfit(Xte_cca0, Yte_cca0, 1)
line = coef[0] * Xte_cca0 + coef[1]
imin = Xte_cca0.argmin()
imax = Xte_cca0.argmax()
x = [Xte_cca0[imin], Xte_cca0[imax]]
y = [line[imin], line[imax]]

# Plot data projected on to first canonical direction
plt.scatter(Xtr_cca0, Ytr_cca0, c='b', label='train')
plt.scatter(Xte_cca0, Yte_cca0, c='r', label='test')
plt.plot(x, y, 'k--', lw=2, label='corr. test')
plt.xlabel('X scores')
plt.ylabel('Y scores')
plt.title('R_tr = {:.2f}\nR_te = {:.2f}'.format(score_tr, score_te))
plt.legend()
plt.show()



cca = CCA(n_components=2)
Xtr_cca, Ytr_cca = cca.fit_transform(Xtr, Ytr)
Xte_cca, Yte_cca = cca.transform(Xte, Yte)

score_tr = cca.score(Xtr, Ytr)
score_te = cca.score(Xte, Yte)
print('Score tr: ', score_tr)
print('Score te: ', score_te)

X_recon, Y_recon = cca.inverse_transform(Xtr_cca[31].reshape(1,-1),
                                         Ytr_cca[31].reshape(1,-1))
print('X recon: ', X_recon)
print('X true:  ', Xtr[31])
print('Y recon: ', Y_recon)
print('Y true:  ', Ytr[31])
