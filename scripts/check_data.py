import argparse
import logging
from pathlib import Path

from dles.utils import set_logger
from dles.utils.data_utils import DataManager

# Check availability of ANI data.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', default=logging.INFO, type=int, help='Logging level.')
    parser.add_argument("-p", "--path", default=None, type=Path, help="Path to data in hdf5-format.")
    args = parser.parse_args()
    set_logger(args.logging)

    ani1_path = args.path
    if not Path.exists(ani1_path):
        raise FileNotFoundError("Please give a valid path to hdf5 data.")
    if ".h5" != str(ani1_path)[-3:]:
        raise ValueError("Please provide data in an h5-format.")

    data = DataManager()
    data.add_path("ani1", ani1_path)
    subset = data.define_subset_ani1(max_num_non_h=2)
