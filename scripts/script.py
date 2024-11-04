import time
import logging
from pathlib import Path

import h5py
import ase
import numpy as np

path_to_data = Path('/Users/philipphuber/Documents/Projects/DLES/Data')
ani1x = Path('ani1x-release.h5')

with h5py.File(path_to_data/ani1x, "r") as h5file:
    nf = 0
    print(len(h5file))
    for num, key1 in enumerate(h5file.keys()):
        a = np.array(h5file[key1]['wb97x_dz.forces'][()])
        for nconf in range(len(a)):
            f_total = 0
            for natom in range(len(a[nconf])):
                f_total = f_total + np.linalg.norm(a[nconf][natom])
            if f_total < 0.005:
                nf = nf + 1
                print(key1)
                print(f'F=0: {num}, {nconf}, {f_total}')
                print(a.shape)
    print(nf)
    print(num)