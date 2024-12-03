import time

import numpy as np
import scipy
from pyscf import scf, df, lib

from dles.hf.molecule import Molecule


def make_mf(m: Molecule = None, dens_fit: bool = False, auxbasis: str = None) -> dict:
    """
    Generates RHF values using pyscf.

    Parameters
    __________
    M : Molecule
        Molecule object.
    dens_fit : bool
        Use density fitting if True.
    auxbasis : str
        Auxilliary basis set.

    Returns
    _______
    qc_parameters : dict
        Dictionary with all relevant QC parameters.
    """
    qc_parameters = {}

    if dens_fit:
        mf = scf.RHF(m).density_fit()
        auxmol = df.addons.make_auxmol(m, auxbasis)
    else:
        mf = scf.hf.RHF(m)
    mf.conv_tol = 1e-12
    mf.max_cycle = 50
    mf.kernel()

    h1e = mf.get_hcore()  # Core Hamiltonian with m.intor_symmetric(), 'int1e_kin' + 'int1e_nuc'
    s1e = mf.get_ovlp()  # Overlap matrix S with intor_symmetric('int1e_ovlp')
    f = mf.get_fock()  # Fock matrix
    eig, mo_coeff = mf.eig(f, s1e)  # Solves HC = SCE with scipy.linalg.eigh(f, s1e)
    dm = mf.make_rdm1()  # Density matrix
    mo_occ = mf.mo_occ  # Electron occupation numbers
    nao = m.nao  # Number of atomic basis functions sum_(cGTO_i) (2l_i+1) (cGTO_i)

    start_time = time.time()
    # get ERI or DF-ERI
    if dens_fit:
        # Calculate integrals
        int3c = df.incore.aux_e2(m, auxmol, 'int3c2e', aosym='s1', comp=1)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        # Calculate df_coeff
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(int2c, int3c.reshape(nao * nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        # Calculate ERI
        df_eri = lib.einsum('ijP,Pkl->ijkl', int3c, df_coef)
        # Calculate Potentials J and K with density fitting
        vj, vk = scf.hf.dot_eri_dm(df_eri, dm, hermi=1, with_j=True, with_k=True)
        # If K without density fitting is required, explicitly add it
        qc_parameters['df_eri'] = df_eri
    else:
        # Calculate ERI
        eri = m.intor("int2e", aosym='s8')
        # Calculate Potentials J and K without density fitting
        vj, vk = scf.hf.dot_eri_dm(eri, dm, hermi=1, with_j=True, with_k=True)
        qc_parameters['eri'] = eri

    end_time = time.time()
    print(f'Time1: {end_time - start_time} s')
    vhf = vj - vk * .5  # HF potential: mf.get_veff(m, dm, hermi=1)
    e_elec = mf.energy_elec(dm, h1e, vhf)  # Electronic energy
    nuc = mf.energy_nuc()  # Nuclear potential
    e_tot = e_elec[0] + nuc  # Total energy

    start_time = time.time()
    eri_ref = m.intor("int2e").reshape((nao, nao, nao, nao))
    vj_check = np.einsum('ijkl,kl->ij', eri_ref, dm)
    vk_check = np.einsum('ilkj,kl->ij', eri_ref, dm)
    end_time = time.time()
    print(f'Time2: {end_time - start_time} s')
    vhf = vj_check - vk_check * .5  # HF potential: mf.get_veff(m, dm, hermi=1)
    e_elec = mf.energy_elec(dm, h1e, vhf)  # Electronic energy

    qc_parameters['e_tot'] = e_tot
    qc_parameters['nuc'] = nuc
    qc_parameters['e_elec'] = e_elec
    qc_parameters['dm'] = dm
    qc_parameters['s1e'] = s1e
    qc_parameters['h1e'] = h1e
    qc_parameters['mo_coeff'] = mo_coeff
    qc_parameters['mo_occ'] = mo_occ

    return qc_parameters


def get_vj(
        m: Molecule = None,
        dm: np.ndarray = None,
        eri: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the Coulomb matrix vj from compact eri and dm arrays.

    Parameters
    __________
    m : Molecule
        The Molecule class.
    eri np.ndarray
        The compact electron repulsion integrals array.
    dm : np.ndarray
        The density matrix in full shape (nao, nao).

    Returns
    _______
    vj : np.ndarray
        The Coulomb matrix vj in full shape (nao, nao).
    """
    nao = m.nao
    nao_pair = nao * (nao + 1) // 2
    vj_compact = np.zeros(nao_pair)
    vj_full = np.zeros((nao, nao))

    # Convert dm to compact representation
    tril_indices = np.tril_indices(nao)
    dm_compact = 2 * dm[tril_indices[0], tril_indices[1]]
    idx = np.arange(nao)
    dm_compact[idx * (idx + 1) // 2 + idx] *= .5

    # Compute compact vj
    for pq in range(nao_pair):
        for rs in range(nao_pair):
            if pq < rs:
                eri_idx = rs * (rs + 1) // 2 + pq
            else:
                eri_idx = pq * (pq + 1) // 2 + rs
            vj_compact[pq] += eri[eri_idx] * dm_compact[rs]

    # Expand vj_compact to full matrix
    for i in range(nao):
        for j in range(i + 1):
            if i < j:
                idx = j * (j + 1) // 2 + i
            else:
                idx = i * (i + 1) // 2 + j
            vj_full[i, j] = vj_compact[idx]
            if i != j:
                vj_full[j, i] = vj_compact[idx]
    return vj_full
