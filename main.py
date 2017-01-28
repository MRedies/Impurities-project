#!/usr/bin/python
import GreenF
import sys
import DensityN
import numpy as np
import matplotlib.pyplot as plt


print("please select variable set:\n")
print("1) m=1,  alpha=0, beta=0, B0=1")
print("2) m=1,  alpha=1, beta=0, B0=0")
print("3) m=1,  alpha=0, beta=1, B0=0")
print("4) m=10, alpha=1, beta=0, B0=0")
print("5) m=10, alpha=1, beta=0, B0=1")

#parameter
if len(sys.argv) == 1:
    i=int(input("set=")) - 1
elif len(sys.argv) == 2:
    i = int(sys.argv[1]) - 1





E = np.linspace(-10, 20, 500)

sets = np.array([
    [1.0,  1e-4, 0.0, 1.0],
    [1.0,  1.0,  0.0, 0.0],
    [1.0,  0.0,  1.0, 0.0],
    [10.0, 1.0,  0.0, 0.0],
    [10.0, 1.0,  0.0, 1.0],
    [1.0,  1.0,  0.0, 1.0]])



a = GreenF.GF(   sets[i,0], sets[i,1], sets[i,2], sets[i,3], eta = 1e-7)
b = DensityN.DOS(sets[i,0], sets[i,1], sets[i,2], sets[i,3])

DOS_G    = np.zeros(E.shape)
DOS_N    = np.zeros(E.shape, dtype=np.complex_)
DOS_Npl  = np.zeros(E.shape, dtype=np.complex_)
DOS_Nmi  = np.zeros(E.shape, dtype=np.complex_)

for j in range(E.shape[0]):
    DOS_G[j]    = a.N(E[j])
    DOS_Npl[j]    = b.Npl(E[j])
    DOS_Nmi[j]    = b.Nmi(E[j])
    DOS_N[j]     = b.N(E[j])
plt.show()

plt.plot(E, DOS_G, linewidth=2,  label='green')
plt.plot(E, np.real(DOS_N),   label="DOS")
plt.plot(E, np.real(DOS_Npl), label='k+ part')
plt.plot(E, np.real(DOS_Nmi), label='k- part')



plt.title("set "+str(i+1)+")  "+"m="+str(sets[i,0])+"; alpha="+str(sets[i,1])+"; beta="+str(sets[i,2])+"; B0="+str(sets[i,3]))
#plt.ylim([-3,7])
plt.legend()
plt.show()

