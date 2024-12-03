import numpy as np

from pyscf import gto, df


class Molecule(gto.Mole):
    """
    Inherits gto.Mole from PySCF. Only some useful attributes inherited from gto.Mole are listed below.

    Attributes
    __________
    atom : list
        Coordinates of atoms.
    basis : str
        Basis set.
    charge : int
        Charge.
    spin : int
        Spin
    nbas : int
        Number of cGTOs
    ao_loc : ndarray
        Continuous ranges for cGTOs as defined in nbas
    natm : int
        Number of atoms
    cart : bool
        Using spherical harmonics (False, default) or cartesian (True)
    _basis : dict
        Basis set for elements used in the molecule.
    _env : ndarray
        GTO exponents (see _bas).
    _bas : ndarray
        [atom number, l, number of primitives, ?, ?,
        start position of GTO exponents saved in _env,
        end position of GTO exponents saved in _env, ?]
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray = None,
        coordinates: np.ndarray = None,
        basis: str = None,
        charge: int = 0,
        spin: int = 0,
    ) -> None:
        """
        Initialize the parent gto.Mole class

        Parameters
        __________
        atomic numbers : np.ndarray
            Atomic numbers of ith atom.
        coordinates : np.ndarray
            Coordinates of ith atom.
        basis : str
            Basis set
        charge : int
            Charge of molecule.
        spin :  int
            Spin of molecule.
        """
        super().__init__()

        symbols = [gto.mole._symbol(number) for number in atomic_numbers]
        atom = [(symbols[i], coordinates[i]) for i in range(len(symbols))]
        self.atom = atom
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.build()

    def make_memory_estimate(self) -> dict:
        """
        Gives an estimation for required memory

        Returns
        _______
        memory : dict
            Memory required for arrays.
        """
        memory = {}
        nao = self.nao
        size_eri = (nao*(nao+1)//2) * (((nao*(nao+1)//2) + 1) // 2)
        m_eri = (size_eri * 8)  # in bytes
        memory['mem_2e'] = m_eri
        m_h1e = nao**2 * 8
        memory['mem_1e'] = m_h1e
        return memory
