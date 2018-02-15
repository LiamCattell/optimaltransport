import glob
import numpy as np
import pandas as pd
from os.path import join
from scipy import misc

from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT


def load_original():
    root = join('..','..','..','data','brain')

    # Get the dataset info
    dataset = pd.read_excel(join(root,'brain_info.xlsx'))

    # Initialise the data and label arrays
    n_imgs = len(dataset['status'])
    # X = np.zeros((n_imgs,109,91))
    X = np.zeros((n_imgs,54,45))
    y = np.zeros(n_imgs, dtype='uint8')

    for i in range(n_imgs):
        print('Image {} of {}'.format(i+1, n_imgs))

        # Load the image and scale by factor of 0.5
        fpath = join(root, 'images', dataset['ptid'][i] + '.png')
        tmp = misc.imread(fpath, flatten=True)
        tmp = misc.imresize(tmp, 0.25)
        tmp = np.rot90(tmp, 2)
        X[i] = signal_to_pdf(tmp.astype(np.float64))

        if dataset['status'][i] == 'p':
            y[i] = 1

    np.savez(join('optrans','datasets','adni_data'), X=X, y=y)
    return


def generate_radoncdt():
    with np.load(join('optrans','datasets','adni_data.npz')) as data:
        X = data['X']
        y = data['y']

    img0 = signal_to_pdf(X.mean(axis=0))

    radoncdt = RadonCDT()
    rcdt = []
    transport_map = []
    displacements = []

    for i,img1 in enumerate(X):
        print('R-CDT {} of {}'.format(i+1, X.shape[0]))
        rcdt.append(radoncdt.forward(img0, img1))
        transport_map.append(radoncdt.transport_map_)
        displacements.append(radoncdt.displacements_)

    rcdt = np.array(rcdt)
    transport_map = np.array(transport_map)
    displacements = np.array(displacements)

    np.save(join('optrans','datasets','adni_img0'), img0)
    np.savez(join('optrans','datasets','adni_rcdt'), rcdt=rcdt, y=y)
    np.savez(join('optrans','datasets','adni_rcdt_maps'), y=y,
             transport_map=transport_map)
    np.savez(join('optrans','datasets','adni_rcdt_disp'), y=y,
             displacements=displacements)

    return


if __name__ == '__main__':
    load_original()
    generate_radoncdt()
