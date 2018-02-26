import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.decomposition import PLDA

space = 'rcdt'
n_splits = 5
random_state = 42

if space == 'image':
    X, y = adni.load_data()
else:
    X, y = adni.load_rcdt()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs, h*w))

print(np.bincount(y))

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
acc = np.zeros(n_splits)

for i,(tr,te) in enumerate(cv.split(X, y)):
    pca = PCA()
    Xtr_pca = pca.fit_transform(X[tr])
    Xte_pca = pca.transform(X[te])

    plda = PLDA(n_components=2, alpha=5.)
    plda.fit(Xtr_pca, y[tr])
    acc[i] = plda.score(Xte_pca, y[te])
    print('Fold ', i, ' -- acc: ', acc[i])

print('ACC: ', acc.mean())
print('STD: ', acc.std())

# Transform final split
Xte_plda = plda.transform(Xte_pca)

# Create a grid of points to plot the shaded regions
xx, yy = np.meshgrid(np.linspace(1.05*Xte_plda[:,0].min(),1.05*Xte_plda[:,0].max(),300),
                     np.linspace(1.05*Xte_plda[:,1].min(),1.05*Xte_plda[:,1].max(),300))

grid_pca = plda.inverse_transform(np.c_[xx.ravel(),yy.ravel()])

# Get predicted classes of grid points
zz = plda.predict(grid_pca).reshape(xx.shape)

print(plda.coef_)
print(plda.intercept_)
print(plda.class_means_)

cols = ['royalblue', 'red']
names = ['Healthy', 'AD']

# Plot test data
fig, ax = plt.subplots(1, 1)
ax.contourf(xx, yy, zz, cmap=colors.ListedColormap(cols), alpha=0.2)
for lab,c,name in zip([0,1],cols,names):
    ax.scatter(Xte_plda[y[te]==lab,0], Xte_plda[y[te]==lab,1], c=c, s=40,
               alpha=1., label=name)
ax.set_title('Accuracy = {:.1f}%'.format(acc[i]*100))
plt.legend(fontsize=12, loc=2)
plt.show()
