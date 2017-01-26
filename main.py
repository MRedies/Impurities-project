#!/usr/bin/python
import GreenF
import DensityN
import numpy as np
import matplotlib.pyplot as plt


print("please select variable set:\n")
print("0) m=1,  alpha=0, beta=0, B0=1")
print("1) m=1,  alpha=1, beta=0, B0=0")
print("2) m=1,  alpha=0, beta=1, B0=0")
print("3) m=10, alpha=1, beta=0, B0=0")
print("4) m=10, alpha=1, beta=0, B0=1")

#parameter
i=int(input("set="))





E = np.linspace(-2, 8, 50)
sets = np.array([[1.0,1e-4,0.0,1.0],  
                 [1.0,1.0,0.0,0.0],
                 [1.0,0.0,1.0,0.0], 
                 [10.0,1.0,0.0,0.0],
                 [10.0,1.0,0.0,1.0],
                 [1.0, 0.0, 1e-4, 1.0]])



a = GreenF.GF(   sets[i,0], sets[i,1], sets[i,2], sets[i,3])
b = DensityN.DOS(sets[i,0], sets[i,1], sets[i,2], sets[i,3])

DOS_G    = np.zeros(E.shape)
DOS_G_SA = np.zeros(E.shape)
DOS_N    = np.zeros(E.shape)
DOS_N2   = np.zeros(E.shape)
DOS_N3   = np.zeros(E.shape)

for j in range(E.shape[0]):
    DOS_G[j]    = a.N(E[j])
   # DOS_G_SA[j] = a.N_small_angle(E[j])
    DOS_N[j]    = b.N(E[j])
    DOS_N2[j]   = b.N2(E[j])
    DOS_N3[j]   = b.N3(E[j])

plt.plot(E,DOS_G,label='green')
plt.plot(E,DOS_N,label="check")
plt.plot(E,DOS_N3,  label="check 3")

plt.title("m="+str(sets[i,0])+"; alpha="+str(sets[i,1])+"; beta="+str(sets[i,2])+"; B0="+str(sets[i,3]))
#plt.ylim([-3,7])
plt.legend()
plt.show()
