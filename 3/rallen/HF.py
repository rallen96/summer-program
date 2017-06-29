import psi4
import numpy as np
import scipy.linalg as sc
import configparser
	
class HF:

	def __init__(self, filename = 'N2.ini'):
		config 	    = configparser.ConfigParser()
		config.read(filename)
		self.mol    = psi4.geometry(config['DEFAULT']['molecule'])
		self.mol.update_geometry()
		basis  	    = psi4.core.BasisSet.build(self.mol, 'BASIS', config['DEFAULT']['basis'])		
		self.mints  = psi4.core.MintsHelper(basis)
		

		self.maxiter= int(config['SCF']['max_iter'])
		self.e_convergence = 1.e-10

		self.E 	    = 0.0								# sets inital E to 0
		self.Vnu    = self.mol.nuclear_repulsion_energy()				# Nuc replusion E
		self.T 	    = self.mints.ao_kinetic().to_array()				# gives 1 e- kinetic integral
		self.S 	    = self.mints.ao_overlap().to_array()				# gives 1 e- overlap integral
		self.V	    = self.mints.ao_potential().to_array()				# gives 1 e- potential energy integral
		self.g	    = self.mints.ao_eri().to_array()					# gives 2 e- integrals
		self.h	    = self.T + self.V							# declares one electron hamiltonian
		self.C 	    = np.zeros_like(self.h)
		self.e	    = np.zeros(len(self.h))						

		self.nelec  = -self.mol.molecular_charge()					# accounts for e- in ions						
		for n in range(self.mol.natom()):
			self.nelec += int(self.mol.Z(n))					# adds electrons based on Z values of atoms
		self.nocc   = int(self.nelec/2)

		if self.mol.multiplicity() != 1 or self.nelec % 2:
			raise Exception("RHF is only for closed shell molecules")		# stops the program if the system is open shell

		
		self.nbf    = self.mints.basisset().nbf()					# spatial orbitals/number of basis functions
		self.D	    = np.zeros_like(self.h)						# initializes D matrix to the nxn zero matrix where n=nbf
		self.X 	    = np.matrix(sc.inv(sc.sqrtm(self.S)))				# builds orthogonalizer 
		self.nu     = np.matrix(np.zeros_like(self.h))					

	def computeEnergy(self):
		Eold = 0.0													
		Dold = np.zeros_like(self.h)
		g, h, X, mol, maxiter, nocc, Vnu = self.g, self.h, self.X, self.mol, self.maxiter, self.nocc, self.Vnu
		for i in range(1, maxiter+1):											
			J = np.einsum('pqrs, rs->pq', g, Dold)              	   						# forms coulomb energy from 2 e- integrals and density matrix
		    	K = np.einsum('prqs, rs->pq', g, Dold)                 							# forms exchange energy from ""
		    	F = h + J*2 - K                                     							# forms fock matrix from 1 e- hamiltonian, coulomb E, exchange
		    	Ft = X.dot(F).dot(X)                                  							# orthogonalizes fock matrix to create eigenvalue problem
		    	e, C = np.linalg.eigh(Ft)                              							# forms eigenvector e and Ctilde from orthogonalized fock matrix
		    	C = X.dot(C)                                            						# converts C back to original form	
		    	Cocc = C[:,:nocc]											# removes unoccupied orbitals from coefficient matrix
		    	D = np.einsum('pi, qi->pq', Cocc, Cocc)     	          						# builds new density matrix 
		    
		    	E = np.einsum('pq, pq->', F+h, D) + Vnu  								# forms new E from Fock matrix, 1 e- hamiltonian, Density matrix,Vnuc	
			deltaE = E - Eold											# calculates energy difference between each iteration
			print('RHF iteration {:3d}: energy {:20.14f} DeltaE {:2.5E}'.format(i, E, deltaE))				
			
			if np.fabs(deltaE) < self.e_convergence: 								# checks for convergence
				break
			Eold = E
			Dold = D

		self.E = E
		
											#test
if __name__=='__main__': 

    rhf = HF('N2.ini')
    rhf.computeEnergy()




		

		








