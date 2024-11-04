import numpy as np
import pyscf.scf.hf
from pyscf import scf


class RHF(pyscf.scf.hf.RHF):
    """
    Class handling the RHF method inherited from pyscf.scf.hf.RHF.
    """

    def __init__(self, M = None):
        """
        Initialize RHF setup.
        """
        if M == None:
            print("No molecule define!")
        super().__init__(M)


    def make_mf(self, M):
        """
        Generates RHF values using pyscf.

        """
        eri = M.intor("int2e", aosym='s8')  # electron repulsion energy
        ke = M.intor("int1e_kin")  # kinetic energy
        nae = M.intor("int1e_nuc")  # nuclear attraction energy

        mf = scf.hf.RHF(M)
        #mf.conv_tol = 1e-12
        mf.max_cycle = 50
        mf.kernel()

        init_guess = mf.get_init_guess()
        #  Initial guess for HF
        h1e = mf.get_hcore()
        #  Calculates core Hamiltonian
        #  Integrals are assumed to be hermitian
        #  h = mol.intor_symmetric('int1e_kin')
        #  h+= mol.intor_symmetric('int1e_nuc')
        #  functions call intor(intor, comp=None, hermi=1, aosym='s4', grids=None) -> getints -> getints2c
        s1e = mf.get_ovlp()
        #  Calculates overlap matrix S
        #  intor_symmetric('int1e_ovlp')
        #  intor(intor, comp=None, hermi=1, aosym='s4', grids=None) -> getints -> getints2c
        f = mf.get_fock()
        eig, mo_coeff = mf.eig(f, s1e)
        #  Solves HC = SCE
        #  e, c = scipy.linalg.eigh(f, s1e)
        #  idx = numpy.argmax(abs(c.real), axis=0)
        #  c[:, c[idx, numpy.arange(len(e))].real < 0] *= -1
        occ = mf.get_occ()
        #  Electron occupation numbers
        #  sum(occ)=Ne, dim(occ) = nbas
        dm = mf.make_rdm1()
        #  mocc = mo_coeff[:, mo_occ > 0]
        #  dm = (mocc * mo_occ[mo_occ > 0]).dot(mocc.conj().T)
        vhf = mf.get_veff(self, dm, dm_last=None, vhf_last=None, hermi=1)
        #  HF potential
        #  calls scf.hf.RHF.get_jk
        #  calls scf.hf.dot_eri_dm
        #  vhf = vj - vk * .5
        nuc = mf.energy_nuc()

        e_elec = mf.energy_elec(dm, h1e, vhf)
        # electronic energy
        #  e1 = numpy.einsum('ij,ji->', h1e, dm).real
        #  e_coul = numpy.einsum('ij,ji->', vhf, dm).real * .5
        #  e_tot = e1+e_coul
        e_tot = e_elec[0] + nuc
        vj, vk = scf.hf.dot_eri_dm(eri, dm, hermi=1, with_j=True, with_k=True)
        vhf = vj - vk * .5
        e_elec = mf.energy_elec(dm, h1e, vhf)
        e_tot = e_elec[0] + nuc
        print(e_tot, e_elec, nuc)
        print(np.einsum('ij,ji->', h1e, dm).real)
        print(np.einsum('ij,ji->', vhf, dm).real * .5)
        print(np.einsum('ij,ji->', ke, dm).real)
        print("SCF Convergence Criterion (Energy):", mf.conv_tol)
        print("SCF Convergence Criterion (Density Matrix):", mf.conv_tol_grad)