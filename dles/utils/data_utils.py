import logging
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import ase
from ase import data
from ase import Atoms
from ase.io import write


class DataManager:
    """
    Manages Data structure.

    Attributes
    __________
    paths : dict
        Paths to data.
    """

    def __init__(self):
        """
        Parameters
        __________
        paths : dict
            Paths to data.
        """

        self.paths = {}

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

    def explore_ani1(self) -> None:
        """
        Exploration of data. Makes some statistics.
        """
        with h5py.File(self.paths["ani1"], "r") as h5file:
            maxan = 0
            for key in h5file.keys():
                maxan = max(max(np.array(h5file[key]["atomic_numbers"])), maxan)
            for num1, key1 in enumerate(h5file.keys()):
                print(key1)
                c = np.array(h5file[key1]["coordinates"])
                atomic_numbers = np.array(h5file[key1]["atomic_numbers"])
                number_of_atoms = len(atomic_numbers)
                print(number_of_atoms, c.shape[0])

    def define_subset_ani1(self, max_num_non_h) -> list:
        """
        Exploration of data. Makes some statistics.

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


