import sys
sys.path.append('../../optimaltransport')

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
from os.path import join

from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT


def load_original():
    root = 'X:/Data/MRI/oasis_cross-sectional'
    dataset = pd.read_excel(join(root,'oasis_cross-sectional.xlsx'))

    # Only get session 1 data
    dataset = dataset[dataset['Session']==1]

    pid = dataset['ID'].tolist()

    y = np.array([0 if row=='hc' else 1 for row in dataset['Dx'].tolist()])

    sex = [0 if row=='F' else 1 for row in dataset['M/F'].tolist()]
    age = dataset['Age'].tolist()
    edu = dataset['Educ'].tolist()
    ses = dataset['SES'].tolist()
    mmse = dataset['MMSE'].tolist()
    cdr = dataset['CDR'].tolist()
    etiv = dataset['eTIV'].tolist()
    nwbv = dataset['nWBV'].tolist()
    asf = dataset['ASF'].tolist()
    metadata = np.array([sex, age, edu, ses, mmse, cdr, etiv, nwbv, asf]).T

    n_samples = len(dataset)
    X = np.zeros((n_samples,208,176))

    for i,p in enumerate(pid):
        print(i, p)
        fname = '*111_t88_masked_gfc.img'
        fpath = glob.glob(join(root, p, 'PROCESSED', 'MPRAGE', 'T88_111', fname))[0]

        vol = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
        print(vol[85].shape)
        tmp = np.rot90(vol[85], 2)
        X[i] = signal_to_pdf(tmp.astype(np.float64), total=1000.)

    np.savez(join('..', 'optrans','datasets','oasis_data'),
             X=X, y=y, metadata=metadata)
    return


def generate_radoncdt():
    with np.load(join('..', 'optrans','datasets','oasis_data.npz')) as data:
        X = data['X']
        y = data['y']
        metadata = data['metadata']

    img0 = signal_to_pdf(X.mean(axis=0), total=1000.)

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

    np.save(join('..', 'optrans','datasets','oasis_img0'), img0)
    np.savez(join('..', 'optrans','datasets','oasis_rcdt'), rcdt=rcdt, y=y,
             metadata=metadata)
    np.savez(join('..', 'optrans','datasets','oasis_rcdt_maps'), y=y,
             transport_map=transport_map, metadata=metadata)
    np.savez(join('..', 'optrans','datasets','oasis_rcdt_disp'), y=y,
             displacements=displacements, metadata=metadata)

    return



if __name__ == '__main__':
    load_original()
    generate_radoncdt()
