import argparse
import logging
from pathlib import Path

from dles.utils import set_logger
from dles.utils.data_utils import DataManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', default=logging.INFO, type=int, help='Logging level.')

    args = parser.parse_args()
    set_logger(args.logging)

    ani1_path = Path("/Users/philipphuber/Documents/Projects/DLES/Data/ani1x.h5")
    data = DataManager()
    data.add_path("ani1", ani1_path)
    subset = data.define_subset_ani1(2)
    print(subset)