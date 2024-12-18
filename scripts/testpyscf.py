import argparse
import logging
import time
from pathlib import Path

from dles.utils import set_logger
from dles.utils.data_utils import DataManager
from dles.hf.molecule import Molecule
from dles.hf.rhf_utils import make_mf

# Script to explore and test ANI data.

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
    data.get_keys()

    subset = data.define_subset_ani1(max_num_non_h=2)
    data.explore_ani1(subset, print_all=True)
    data.memory_estimation(subset)

    do_scf = True
    if do_scf:
        for key in ['O2']:
            start_time = time.time()
            print(key)
            atomic_numbers, coordinates = data.get_ani1_data(key)
            m = Molecule(atomic_numbers, coordinates[0, :, :], basis='def2svp')
            auxbasis = 'def2-svp-jfit'  # Only j with ri, k with 4c integrals
            qc_parameters = make_mf(m, dens_fit=False, auxbasis=auxbasis)
            #DataManager.write_xyz(data, key=key, num_conf=[0], file_format='coord')
            end_time = time.time()
            print(f'Time: {end_time - start_time} s')
