import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import VOT2D
from optrans.datasets import adni
from optrans.utils import interp2d

X, _ = adni.load_data()
img0 = adni.load_img0()

h, w = img0.shape
x, y = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

f = np.stack((y,x), axis=0)
f[0] += 10.3

img0f = interp2d(img0, f)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img0)
ax[1].imshow(img0f)
plt.show()

# vot = VOT2D()
#
# vot.forward(img0, X[1])
