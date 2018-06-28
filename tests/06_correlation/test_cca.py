import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA as CanonCorr
from sklearn.model_selection import train_test_split

from optrans.datasets import oasis
from optrans.decomposition import CCA, fit_line

"""
Perform CCA on the OASIS brain MRI dataset metadata. Plots the data projected
on to the first canonical direction.

Liam Cattell -- May 2018
"""

# Get the OASIS metadata and discard any rows containing NaNs
_, _, metadata = oasis.load_data()
metadata = metadata[~np.isnan(metadata).any(axis=1)]

# Select the metadata to use
X = metadata[:,[1,4]]       # [age, mmse]
Y = metadata[:,6:9]       # [etiv, nwbv, asf]

# Split into training and testing data
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.25, random_state=42)

# Perform CCA
cca = CCA(n_components=2)
Xtr_cca, Ytr_cca = cca.fit_transform(Xtr, Ytr)
Xte_cca, Yte_cca = cca.transform(Xte, Yte)

# Get the correlation coefficients for the training and testing data
score_tr = cca.score(Xtr, Ytr)
score_te = cca.score(Xte, Yte)
print('Correlation\n-----------')
print('Score tr: ', score_tr)
print('Score te: ', score_te)

# Try reconstructing some data using inverse_transform()
X_recon, Y_recon = cca.inverse_transform(Xtr_cca[31].reshape(1,-1),
                                         Ytr_cca[31].reshape(1,-1))
print('\nReconstructing data\n-------------------')
print('X true:  ', Xtr[31])
print('X recon: ', X_recon)
print('Y true:  ', Ytr[31])
print('Y recon: ', Y_recon)

# Get correlation line
xl, yl = fit_line(Xte_cca[:,0], Yte_cca[:,0])

# Plot correlation results
plt.figure()
plt.scatter(Xtr_cca[:,0], Ytr_cca[:,0], c='b', label='train')
plt.scatter(Xte_cca[:,0], Yte_cca[:,0], c='r', label='test')
plt.plot(xl, yl, 'k--', lw=2, label='corr. test')
plt.xlabel('X scores')
plt.ylabel('Y scores')
plt.title('1st CCA direction\nR_tr = {:.2f}, R_te = {:.2f}'.format(score_tr[0],
                                                                   score_te[0]))
plt.legend()
plt.show()
