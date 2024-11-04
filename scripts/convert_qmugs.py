import argparse
import logging
from pathlib import Path

import ase
import ase.io
import h5py

from dles.utils import set_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', default=logging.INFO, type=int, help='Logging level.')

    args = parser.parse_args()
    set_logger(args.logging)

    qmugs_sdf_path = Path("/Users/philipphuber/Documents/Projects/DLES/Data/qmugs/structures")
    qmugs_hdf_path = Path("/Users/philipphuber/Documents/Projects/DLES/Data/qmugs/qmugs.hdf5")
    with h5py.File(qmugs_hdf_path, "r") as h5file:
        for num1, entry1 in enumerate(qmugs_sdf_path.iterdir()):
            path1 = qmugs_sdf_path / entry1
            if num1 == 5: break
            for num2, entry2 in enumerate(path1.iterdir()):
                path2 = path1 / entry2
                ase.io.read(path2)
                print(path2)

