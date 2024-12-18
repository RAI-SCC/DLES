import argparse
import logging
from pathlib import Path

import ase
import ase.io
import h5py

from dles.utils import set_logger

# Reads data from source. Code for saving data as hdf5 is not written yet.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', default=logging.INFO, type=int, help='Logging level.')
    parser.add_argument("-sp", "--source_path", default=None, type=Path, help="Source of Data")
    parser.add_argument("-tp", "--target_path", default=None, type=Path, help="File to be saved as hdf5.")
    args = parser.parse_args()
    set_logger(args.logging)

    source_directory = args.source_path
    target_path = args.target_path
    if not Path.exists(source_directory):
        raise FileNotFoundError("Please give a valid path to data.")
    if ".h5" != str(target_path)[-3:]:
        raise ValueError("Please save data in an h5-format.")

    with h5py.File(target_path, "w") as h5file:
        for formula in source_directory.iterdir():
            source_path = source_directory / formula
            for entry in source_path.iterdir():
                path = source_path / entry
                atoms = ase.io.read(path)
