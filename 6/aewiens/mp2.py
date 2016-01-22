import numpy as np
import sys
sys.path.insert(0,"../../5/aewiens/")

from uhf import UHF

class UMP2:
    
    def __init__(self,mol,mints):

        uhf = UHF(mol,mints)
        uhf.get_energy()
        self.Ecorr = 0.0


    def get_mp2(self):
        """
        Spin-orbital implementation of mp2 equations
        """

        ##  Get relevant varialbes from UHF object
        nocc, norb = uhf.nocc, uhf.norb
        E0, e, g, C = uhf.E0, uhf.e, uhf.g, uhf.norb


        ####  4 different algorithms for integral transformation (decreasing efficiency) 
        #Gmo = self.tei(g, C, norb)
        #Gmo = self.tei_noddy(g, C, norb)

        #Gmo = self.tei_einsum_noddy(g, C)
        Gmo = self.tei_einsum(uhf.g, uhf.C)

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nocc,norb):
                    for b in range(nocc,norb):
                        self.Ecorr += 0.25 * (Gmo[i,j,a,b])**2 / (e[i] + e[j] - e[a] - e[b])
        return E0 + self.Ecorr


"""
Integral transformation functions: gAO --> gMO using expansion coefficients C
    - 2 different algorithms
    - ^ with and without einsum
"""

def tei(g, C, norb):

    Gmo1 = np.zeros(g.shape)
    for P in range(norb):
        for Q in range(norb):
            for R in range(norb):
                for S in range(norb):
                    for s in range(norb):
                        Gmo1[P,Q,R,s] += C[S,s] * g[P,Q,R,S] 

    Gmo2 = np.zeros(g.shape)
    for P in range(norb):
        for Q in range(norb):
            for R in range(norb):
                for s in range(norb):
                    for r in range(norb):
                        Gmo2[P,Q,r,s] += C[R,r] * Gmo1[P,Q,R,s] 

    Gmo3 = np.zeros(g.shape)
    for P in range(norb):
        for Q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    for q in range(norb):
                        Gmo3[P,q,r,s] += C[Q,q] * Gmo2[P,Q,r,s] 

    Gmo = np.zeros(g.shape)
    for P in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    for p in range(norb):
                        Gmo[p,q,r,s] += C[P,p] * Gmo3[P,q,r,s] 

    return Gmo


def tei_einsum(g,C):
    return np.einsum("Pp,Pqrs->pqrs",C,
                np.einsum("Qq,PQrs->Pqrs",C,
                    np.einsum("Rr,PQRs->PQrs",C,
                        np.einsum("Ss,PQRS->PQRs",C,g))))


def tei_noddy(g,C,norb):
    Gmo = np.zeros(g.shape)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    for P in range(norb):
                        for Q in range(norb):
                            for R in range(norb):
                                for S in range(norb):
                                    Gmo[p,q,r,s] += C[P,p]*C[Q,q]*C[R,r]*C[S,s] * g[P,Q,R,S]
    return Gmo


def tei_noddy_einsum(g,C):
    return np.einsum("PQRS,Pp,Qq,Rr,Ss->pqrs",g,C,C,C,C)