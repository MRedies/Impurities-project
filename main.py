#!/usr/bin/python
import GreenF
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
i=int(input("set="))-1





E = np.linspace(-8, 8, 50)
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
DOS_Npl  = np.zeros(E.shape)
DOS_Nmi  = np.zeros(E.shape)
DOS_N2   = np.zeros(E.shape)
DOS_N3   = np.zeros(E.shape)

for j in range(E.shape[0]):
    DOS_G[j]    = a.N(E[j])
   # DOS_G_SA[j] = a.N_small_angle(E[j])
    DOS_Npl[j]    = b.Npl(E[j])
    DOS_Nmi[j]    = b.Nmi(E[j])
    DOS_N[j]     = b.N(E[j])
    #DOS_N2[j]   = b.N2(E[j])
    #DOS_N3[j]   = b.N3(E[j])

#plt.plot(E,DOS_G,label='green')
plt.plot(E,DOS_N,label="DOS")
plt.plot(E,DOS_Npl,label='k+ part')
plt.plot(E,DOS_Nmi,label='k- part')


#plt.plot(E,DOS_N3,  label="check 3")

plt.title("set "+str(i+1)+")  "+"m="+str(sets[i,0])+"; alpha="+str(sets[i,1])+"; beta="+str(sets[i,2])+"; B0="+str(sets[i,3]))
#plt.ylim([-3,7])
plt.legend()
plt.show()


'''
#plot k+ and k-
kpl = np.zeros(E.shape)
kmi = np.zeros(E.shape)
for j in range(E.shape[0]):
    kpl[j] = b.kplus(E[j])
    kmi[j] = b.kminus(E[j])

plt.title('k(E)')
plt.plot(E,kpl,label='+')
plt.plot(E,kmi,label='-')
plt.legend()
plt.show()
'''

