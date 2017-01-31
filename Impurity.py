import numpy as np
import numpy.linalg as la


class Imp:
    def __init__(self, R, V, B, width = 0.05):
        self.R     = R
        self.V     = V
        self.B     = B
        self.width = width

        self.sigma_0 = np.identity(2, dtype=np.complex_)
        self.sigma_x = np.array([[0, 1],[1, 0]], dtype=np.complex_)
        self.sigma_y = np.array([[0, -1j],[1j, 0]], dtype=np.complex_)
        self.sigma_z = np.array([[1, 0],[0, -1]], dtype=np.complex_)
    
    def d_H(self, r):
        result = np.zeros((2,2), dtype=np.complex_)
        
        for i in range(self.R.shape[0]):
            single = np.zeros((2,2), dtype=np.complex_)

            single += self.V[i]   * self.sigma_0
            single += self.B[i,0] * self.sigma_x
            single += self.B[i,1] * self.sigma_y
            single += self.B[i,2] * self.sigma_z

            dR = la.norm(r - self.R[i,:])
            single *= self.delta_lor(dR)

            result += single

        return result

    def delta_lor(self, t):
        return 1.0/np.pi * self.width /(t**2 + self.width**2)

    def delta_gau(self, t):
        return 1.0/(np.sqrt(2.0 * np.pi) * self.width) \
                * np.exp(-t**2 /(2.0 * self.width**2))
