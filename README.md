![DLES Logo](logo.jpeg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)

# Deep Learning Electronic Structure
DLES is a Python package to load and manipulate molecule data using [ASE](https://wiki.fysik.dtu.dk/ase/)  (Atomic Simulation Environment) and to carry out quantum chemistry methods using [PySCF](https://pyscf.org).
It is an ongoing project, which will be further developed for deep learning (DL) applications. 


## Installation

Clone the project https://github.com/RAI-SCC/DLES.git and install it with `pip install -e .`.

## Data Handling

DLES is able to
- read chemical structures data from HDF5 files.
- save specific structures as coord- or xyz-files using [ASE](https://wiki.fysik.dtu.dk/ase/).
- make memory estimation for matrices generated during Hartree-Fock computations.

## Hartree-Fock Computations
DLES imports [PySCF](https://pyscf.org) to systematically carry out restricted Hartree-Fock computations.

## Deep Learning
DL applications are not implemented yet. Our aim is learning the electronic structure of molecules.