import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from optrans.datasets import adni
from optrans.decomposition import PLDA

space = 'rcdt'
n_splits = 5
random_state = 11

if space == 'image':
    X, y = adni.load_data()
else:
    X, y = adni.load_rcdt()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs, h*w))

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

for i,(tr,te) in enumerate(cv.split(X, y)):
    pca = PCA()
    Xtr_pca = pca.fit_transform(X[tr])
    Xte_pca = pca.transform(X[te])

    plda = PLDA(n_components=2, alpha=1.)
    plda.fit(Xtr_pca, y[tr])
    Xte_plda = plda.transform(Xte_pca)

    # y_pred = plda.predict(Xte_pca)
    y_pred = plda.predict_transformed(Xte_plda)
    acc = accuracy_score(y[te], y_pred)

    print('Fold ', i, ' -- acc: ', acc)
    # print('........ -- ac2: ', acc2)


# Create a grid of points to plot the shaded regions
xx, yy = np.meshgrid(np.linspace(1.05*Xte_plda[:,0].min(),1.05*Xte_plda[:,0].max(),300),
                     np.linspace(1.05*Xte_plda[:,1].min(),1.05*Xte_plda[:,1].max(),300))

# Classify grid in PLDA space
zz = plda.predict_transformed(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)


cols = ['royalblue', 'red']
names = ['Healthy', 'AD']

# Plot test data
fig, ax = plt.subplots(1, 1)

ax.contourf(xx, yy, zz, cmap=colors.ListedColormap(cols), alpha=0.2)
ax.scatter(Xte_plda[:,0], Xte_plda[:,1], c=y[te], s=40,
              cmap=colors.ListedColormap(cols), alpha=0.6)
ax.scatter(Xte_plda[:,0], Xte_plda[:,1], c=y_pred, s=40, marker='x',
              cmap=colors.ListedColormap(cols), alpha=1.)
ax.set_title('Accuracy = {:.1f}%'.format(acc*100))

plt.show()
