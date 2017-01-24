import numpy as np


class DOS:
	def __init__(self,m,alpha,beta,B0):
		self.m      = complex(m,     0)
   		self.alpha  = complex(alpha, 0)
   		self.beta   = complex(beta,  0)
   		self.B0     = complex(B0,    0)
   		self.alpBet = self.alpha**2 + self.beta**2
   		self.E_so   = self.m * (self.alpha**2 + self.beta**2)



   	def kminus(self,E):
   		return np.sqrt(	2*self.m*(   E + self.E_so - np.sqrt( (E+self.E_so)**2+self.B0**2-E**2)  )			)

	def kplus(self,E):
		return np.sqrt(	2*self.m*(   E + self.E_so + np.sqrt( (E+self.E_so)**2+self.B0**2-E**2)  )			)


	
	
	def NPlusKp(self,E):
		return 1./ np.abs(	1./self.m -  self.alpBet/np.sqrt(self.alpBet*self.kplus(E)**2+self.B0**2)	)
	
	def NMinusKp(self,E):
		return 1./ np.abs(	1./self.m + self.alpBet/np.sqrt(self.alpBet*self.kplus(E)**2+self.B0**2)	)
	


	def NPlusKm(self,E):
		return 1./ np.abs(	1./self.m - self.alpBet/np.sqrt(self.alpBet*self.kminus(E)**2+self.B0**2)	)
	
	def NMinusKm(self,E):
		return 1./ np.abs(	1./self.m + self.alpBet/np.sqrt(self.alpBet*self.kminus(E)**2+self.B0**2)	)


	
	def N(self,E):
		#kp = self.kplus(E).real
		#km = self.kminus(E).real
		kp = 0.5 * ( np.sign(self.kplus(E).real)  + 1)
		km = 0.5 * ( np.sign(self.kminus(E).real) + 1)
		return (	kp*self.NPlusKp(E)  + kp*self.NMinusKp(E)		+ 		km*self.NPlusKm(E) + km*self.NMinusKm(E)	) /(2*np.pi)
	


