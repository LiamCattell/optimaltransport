import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT
from optrans.datasets import gaussians

"""
Generate the normalized images and Radon-CDTs for the Gaussian blob dataset.
The outputs are used in the datasets module.

Liam Cattell -- January 2018
"""

def generate_image():
    X, y = gaussians.make_gaussians(n_samples=100, n_dots=[1,2,3])
    for i in range(X.shape[0]):
        X[i] = signal_to_pdf(X[i])

    np.savez(join('optrans','datasets','gaussians_data'), X=X, y=y)
    return


def generate_radoncdt():
    X, y = gaussians.load_data()

    n_imgs, h, w = X.shape
    img0 = signal_to_pdf(np.ones((h,w)))

    radoncdt = RadonCDT()
    rcdt = []
    transport_map = []
    displacements = []

    for i,img1 in enumerate(X):
        print('R-CDT {} of {}'.format(i+1, n_imgs))
        rcdt.append(radoncdt.forward(img0, img1))
        transport_map.append(radoncdt.transport_map_)
        displacements.append(radoncdt.displacements_)

    rcdt = np.array(rcdt)
    transport_map = np.array(transport_map)
    displacements = np.array(displacements)

    np.save(join('optrans','datasets','gaussians_img0'), img0)
    np.savez(join('optrans','datasets','gaussians_rcdt'), rcdt=rcdt, y=y)
    np.savez(join('optrans','datasets','gaussians_rcdt_maps'), y=y,
             transport_map=transport_map)
    np.savez(join('optrans','datasets','gaussians_rcdt_disp'), y=y,
             displacements=displacements)

    return


if __name__ == '__main__':
    generate_image()
    generate_radoncdt()
