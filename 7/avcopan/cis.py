import psi4
import numpy as np
import scipy.linalg as la
import itertools as it
import sys
sys.path.append("../../5/avcopan/")
from uhf import UHF


class CIS:

    def __init__(self, mol, mints):
      uhf = UHF(mol, mints)
      uhf.compute_energy()
      self.nocc = uhf.nocc
      self.nbf  = uhf.nbf
      self.g    = transform_tei(uhf.g, uhf.C) # antisymmetrized two-electron integrals, spin-orbital (MO) basis

    def singles_iterator(self):
      nocc, nbf = self.nocc, self.nbf
      return enumerate(it.product(range(0,nocc), range(nocc,nbf)))

    def compute_excitation_energies(self):
      for P, (i,a) in self.singles_iterator():
        print('{:d} {:d},{:d}'.format(P, i, a))


def transform_tei(gao, C):
  return np.einsum('Pp,Pqrs->pqrs', C, 
           np.einsum('Qq,PQrs->Pqrs', C,
             np.einsum('Rr,PQRs->PQrs', C,
               np.einsum('Ss,PQRS->PQRs', C, gao)
             )
           )
         )
