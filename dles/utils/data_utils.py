import logging
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import ase
from ase import Atoms
from ase.io import write

from dles.hf.molecule import Molecule


class DataManager:
    """
    Manages Data structure.

    Attributes
    __________
    paths : dict
        Paths to data.
    keys : list
        Molecular formulas / keys of dataset
    """

    def __init__(self):
        """
        Parameters
        __________
        paths : dict
            Paths to data.
        """

        self.paths = {}
        self.keys = []

    def add_path(self, name: str, path: Path) -> None:
        """
        Adds paths to dataset.

        Parameters
        __________
        name : str
            Name of Dataset
        path : Path
            Path to dataset
        """
        datasets = ["ani1"]
        if str(name) in datasets:
            self.paths[name] = path

    def get_keys(self) -> None:
        """
        Get molecular formulas / keys of dataset.
        """
        self.keys = []
        with h5py.File(self.paths["ani1"], "r") as h5file:
            for key in h5file.keys():
                self.keys.append(str(key))

    def explore_ani1(self, dataset: list = None, print_all: bool = False) -> None:
        """
        Exploration of data. Makes some statistics.

        Parameters
        __________
        dataset : list
            List of keys / molecular formulas of dataset.
        print_all : bool
            Print information for all keys if True.
        """

        with h5py.File(self.paths["ani1"], "r") as h5file:
            total_num_structures = 0
            for key in dataset:
                c = np.array(h5file[key]["coordinates"])
                atomic_numbers = np.array(h5file[key]["atomic_numbers"])
                total_num_structures = total_num_structures + c.shape[0]
                if print_all:
                    print(f'Formula: {key}')
                    print(f'Number of atoms: {len(atomic_numbers)}')
                    print(f'Number of configurations: {c.shape[0]}')
                    print(f'{40*"_"}')
            print(f'Total number of configurations: {total_num_structures}')
            print(f'Total number of molecular formulas: {len(dataset)}')
            print(40 * '*')

    def memory_estimation(self, dataset: list = None) -> None:
        """
        Estimates required memory.

        Parameters
        dataset : list
            List of keys / molecular formulas of dataset.
        __________
        """
        m_4c_total = 0
        m_2c_total = 0
        for key in dataset:
            atomic_numbers, coordinates = self.get_ani1_data(key)
            m = Molecule(atomic_numbers, coordinates[0, :, :], basis='def2-SVP')
            num_conf = coordinates.shape[0]
            memory = m.make_memory_estimate()
            m_4c = (memory["mem_2e"] / 1024 ** 3) * num_conf
            m_2c = (memory["mem_1e"] / 1024 ** 3) * num_conf
            m_2c_total = m_2c_total + m_2c
            m_4c_total = m_4c_total + m_4c
        print(f'Total memory needed for 4c integrals: {m_4c_total} GiB')
        print(f'Total memory needed for 2c integrals: {m_2c_total} GiB')
        print(f'Total memory: {m_4c_total + m_2c_total * 4} GiB')

    def define_subset_ani1(self, max_num_non_h) -> list:
        """
        defines a subset of the data to include only structures with a number of non-hydrogen atoms below a specified threshold.

        Parameters
        __________
        max_num_non_h : int
            Maximum number of non hydrogen atoms within the sample.

        Returns
        _______
        subset : list
            List containing the keys of ani1 as string, which define the subset.
        """
        subset = []
        logger = logging.getLogger(__name__)
        logger.info(f'Subset of ani1 defined with molecules of a maximum of {max_num_non_h} non hydrogen atoms')
        with h5py.File(self.paths["ani1"], "r") as h5file:
            for key in h5file.keys():
                without_h = (int(i) for i in (h5file[key]["atomic_numbers"]) if i > 1)
                if len(list(without_h)) <= max_num_non_h:
                    subset.append(key)
        return subset

    def get_ani1_data(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get relevant data for processing.

        Parameters
        __________
        key : str
            Molecule name within the dataset.

        Returns
        _______
        atomic_numbers : np.ndarray
            Atomic numbers of the molecule.
        coordinates : np.ndarray
            Coordinates of the molecule.
        """
        with h5py.File(self.paths["ani1"], "r") as h5file:
            atomic_numbers = np.array(h5file[key]["atomic_numbers"])
            coordinates = np.array(h5file[key]["coordinates"])  # in Angstrom
        return atomic_numbers, coordinates

    def write_xyz(self, key: str = None, num_conf: list = None, file_format: str = 'xyz') -> None:
        """
        Writes all structures/conformations of a specific molecular formular into a xyz file.

        Parameters:
        __________
        key : str
            Molecular formular used as key in the HDF5 file.
        num_conf : list
            Numbers (integers) for configurations to be written into file.
        file_format : str
            File format xyz or coord
        """
        # Check format
        if file_format not in ['xyz', 'coord']:
            raise ValueError("wrong format for writing file.")
        # Create directory for structures
        current_path = Path(os.getcwd())
        if not os.path.isdir(current_path / 'structures'):
            os.mkdir(current_path / 'structures')
        # Write structures
        with h5py.File(self.paths["ani1"], "r") as h5file:
            c = np.array(h5file[key]["coordinates"])
            atomic_numbers = np.array(h5file[key]["atomic_numbers"])
            structures = []
            if num_conf is None:
                num_conf = range(len(c.shape[0]))
            out_path = Path(current_path / 'structures' / key)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            for i in num_conf:
                positions = c[i, :, :]
                element_symbols = [ase.data.chemical_symbols[an] for an in atomic_numbers]
                elements = list(element_symbols)
                structures.append(Atoms(elements, positions))
                if file_format == 'xyz':
                    write(f'{out_path}/{key}_structures.{file_format}', structures)
                elif file_format == 'coord':
                    write(f'{out_path}/{file_format}', structures)
