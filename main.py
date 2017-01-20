import GreenF
import numpy as np

a = GreenF.GF(1.0, 1.0, 0.0, 1.0)
z = 4.0 + 1e-5 * 1j
k1, k2 = a.find_k(z)

print("k1 = %f + %f i, D(k1) = %g"%(np.real(k1), np.imag(k1), np.abs(a.D(z, k1))))
print("k2 = %f + %f i, D(k2) = %g"%(np.real(k2), np.imag(k2), np.abs(a.D(z, k2))))
