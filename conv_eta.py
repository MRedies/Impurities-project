#!/usr/bin/python
import GreenF
import DensityN
import numpy as np
import matplotlib.pyplot as plt


x   = np.logspace(-2, -12, 5)
E   = np.linspace(-2,8,300)
DOS = np.zeros(E.shape)


for eta_in in x:
    a = GreenF.GF(1.0, 0.0, 1e-4, 1.0, eta=eta_in)

    
    for j in range(E.shape[0]):
        DOS[j] = a.N(E[j])
    plt.plot(E, DOS, label="$\eta$ = %g"%(eta_in))
plt.legend()
plt.show()
