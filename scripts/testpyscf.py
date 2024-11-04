import argparse
import logging
from pathlib import Path

import numpy as np

from dles.utils import set_logger
from dles.utils.data_utils import DataManager
from dles.hf.molecule import Molecule
from dles.hf.rhf_utils import RHF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', default=logging.INFO, type=int, help='Logging level.')

    args = parser.parse_args()
    set_logger(args.logging)

    ani1_path = Path("/Users/philipphuber/Documents/Projects/DLES/Data/ani1x.h5")
    data = DataManager()
    data.add_path("ani1", ani1_path)
    subset = data.define_subset_ani1(2)

    #M = Molecule(np.array([1, 1]), np.array([[0, 0, 0], [0.39159116426, 0, 0]]), basis='def2-SVP')
    #mf = RHF(M)
    #mf.make_mf(M)
    for k, key in enumerate(subset):
        if k > 3: break
        print(key)
        atomic_numbers, coordinates = data.get_ani1_data(key)
        M = Molecule(atomic_numbers, coordinates[0, :, :], basis='def2-SVP')
        mf = RHF(M)
        mf.make_mf(M)
        DataManager.write_xyz(data, key=key, num_conf=[0], file_format='coord')
