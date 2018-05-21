import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


def gaussian2d(sigma, mode):
    hw = np.ceil(2.*sigma)
    rng =  np.arange(-hw, hw+1, 1.)
    xv, yv = np.meshgrid(rng, rng)

    rho = 1./(2.*np.pi*sigma**2) * np.exp(-(xv**2 + yv**2)/(2.*sigma**2))

    if mode == 0:
        return rho
    else:
        rhoy = -rho * yv/(sigma**2)
        rhox = -rho * xv/(sigma**2)

        if mode == 1:
            return rhoy, rhox
        else:
            rhoyy = (-rhoy * yv/(sigma**2)) + (-rho/(sigma**2))
            rhoyx = -rhox * yv/(sigma**2)
            rhoxy = -rhoy * xv/(sigma**2)
            rhoxx = (-rhox * xv/(sigma**2)) + (-rho/(sigma**2))
            return rhoyy, rhoyx, rhoxy, rhoxx


sigma = 3.

img = camera().astype(np.float)
rho = gaussian2d(sigma=sigma, mode=0)
rhoy, rhox = gaussian2d(sigma=sigma, mode=1)
rhoyy, rhoyx, rhoxy, rhoxx = gaussian2d(sigma=sigma, mode=2)

filt = [rho, rhoy, rhox, rhoyy, rhoyx, rhoxy, rhoxx]
ifilt = [np.rot90(f,2) for f in filt]

fig1, ax1 = plt.subplots(2, 7)
for i in range(7):
    ax1[0,i].imshow(filt[i])
    ax1[1,i].imshow(ifilt[i])

img1 = [img]
for f in filt:
    img1.append(convolve2d(img, f, mode='same'))

img2 = [img]
img2.append(gaussian_filter(img, sigma, order=0))
img2.append(gaussian_filter(img, sigma, order=[1,0]))
img2.append(gaussian_filter(img, sigma, order=[0,1]))
img2.append(gaussian_filter(img, sigma, order=[2,0]))
img2.append(gaussian_filter(img, sigma, order=[1,1]))
img2.append(gaussian_filter(img, sigma, order=[1,1]))
img2.append(gaussian_filter(img, sigma, order=[0,2]))

fig2, ax2 = plt.subplots(3, 8)
for i in range(8):
    ax2[0,i].imshow(img1[i])
    ax2[1,i].imshow(img2[i])
    ax2[2,i].imshow(img1[i]-img2[i])

iimg1 = [img]
for f in ifilt:
    iimg1.append(convolve2d(img, f, mode='same'))

iimg2 = [img]
iimg2.append(gaussian_filter(img, sigma, order=0))
iimg2.append(np.rot90(gaussian_filter(np.rot90(img,2), sigma, order=[1,0]), 2))
iimg2.append(np.rot90(gaussian_filter(np.rot90(img,2), sigma, order=[0,1]), 2))
iimg2.append(gaussian_filter(img, sigma, order=[2,0]))
iimg2.append(gaussian_filter(img, sigma, order=1))
iimg2.append(gaussian_filter(img, sigma, order=1))
iimg2.append(gaussian_filter(img, sigma, order=[0,2]))

fig3, ax3 = plt.subplots(3, 8)
for i in range(8):
    ax3[0,i].imshow(iimg1[i])
    ax3[1,i].imshow(iimg2[i])
    ax3[2,i].imshow(iimg1[i]-iimg2[i])

plt.show()
