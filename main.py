import GreenF
import numpy as np
import matplotlib.pyplot as plt


E = np.linspace(-10, 10, 300)
for pot in np.arange(0,10, 1):
    a = GreenF.GF(1.0, 10**(-pot), 0.0, 1.0)
    z = 4.0 + 1e-5 * 1j
    k1, k2 = a.find_k(z)

    DOS = np.zeros(E.shape)
    for i in range(E.shape[0]):
        DOS[i] = a.N(E[i])

    plt.plot(E, DOS, label=str(-pot))
plt.legend()
plt.show()
