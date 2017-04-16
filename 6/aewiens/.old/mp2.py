#!/usr/bin/env python3
import psi4, sys, numpy as np, configparser as cfp
sys.path.insert(0,"../../5/aewiens/")
from uhf import UHF

class MP2:

	def __init__(self,options):

		uhf = UHF(options)
		uhf.computeEnergy()

		self.nocc = uhf.nocc
		self.norb = uhf.norb
		self.E0   = uhf.E
		self.e    = uhf.G
		self.C    = uhf.C
		self.G    = uhf.G


	def transformTEI(self,g,C):
		return np.einsum("Pp,Pqrs->pqrs",C,
				np.einsum("Qq,PQrs->Pqrs",C,
				np.einsum("Rr,PQRs->PQrs",C,
				np.einsum("Ss,PQRS->PQRs",C,g))))


	def computeEnergy(self):
		"""
		Spin-orbital implementation of mp2 equations
		"""

		Gmo  = self.transformTEI(self.G, self.C)
		nocc = self.nocc
		norb = self.norb

		e  = self.e
		Ec = 0.0
		for i in range(nocc):
			for j in range(nocc):
				for a in range(nocc,norb):
					for b in range(nocc,norb):
						#Ecorr += (2*Gmo[i,j,a,b]-Gmo[i,j,b,a])*Gmo[i,j,a,b]/(e[i]+e[j]-e[a]-e[b])
						Ec += (Gmo[i,j,a,b]*Gmo[a,b,i,j])/(e[i]+e[j]-e[a]-e[b])

		Ec *= 0.25
						
		"""
		o = slice(0,nocc)
		v = slice(nocc,nocc+norb)
		x = np.newaxis

		D = e[o,x,x,x] + e[x,o,x,x] - e[x,x,v,x] - e[x,x,x,v]
		T = Gmo[o,o,v,v]*Gmo[v,v,o,o] 
		#T =  np.square(Gmo[o,o,v,v])
		#T /= D

		Ecorr = 0.25*np.ndarray.sum(T,axis=(0,1,2,3))
		"""

		return self.E0 + Ec



if __name__ == '__main__':
	
	config = cfp.ConfigParser()
	config.read('Options.ini')

	mp2 = MP2(config)

	print( mp2.computeEnergy() )