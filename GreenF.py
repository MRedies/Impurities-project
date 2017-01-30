import numpy as np
import scipy.special as sf
import numpy.linalg as la
import matplotlib.pyplot as plt

class GF:
    def __init__(self, m, alpha, beta, B0, eta = 1e-5, R_to_0 = 1e-6):
        self.m      = complex(m,     0)
        self.alpha  = complex(alpha, 0)
        self.beta   = complex(beta,  0)
        self.B0     = complex(B0,    0)
        self.eta    = eta
        self.R_to_0 = R_to_0
        self.E_so   = self.m * (alpha**2 + beta**2)
        self.E0     = - (self.E_so**2 + self.B0**2)/(2 * self.E_so)

        self.sigma_0 = np.identity(2)
        self.sigma_x = np.array([[0, 1],[1, 0]])
        self.sigma_y = np.array([[0, -1j],[1j, 0]])
        self.sigma_z = np.array([[1, 0],[0, -1]])


        print("m      = %g + i * %g"%(      np.real(self.m),      np.imag(self.m)))
        print("alpha  = %g + i * %g"%(  np.real(self.alpha),  np.imag(self.alpha)))
        print("beta   = %g + i * %g"%(   np.real(self.beta),   np.imag(self.beta)))
        print("B0     = %g + i * %g"%(     np.real(self.B0),     np.imag(self.B0)))
        print("eta    = %g + i * %g"%(    np.real(self.eta),    np.imag(self.eta)))
        print("R_to_0 = %g + i * %g"%( np.real(self.R_to_0), np.imag(self.R_to_0)))
        print("E_so   = %g + i * %g"%(   np.real(self.E_so),   np.imag(self.E_so)))
        print("E_0    = %g + i * %g"%(    np.real(self.E0),     np.imag(self.E0)))

    def find_k(self, z):
        E   = np.real(z)
        eta = np.imag(z)
        inner_sqrt   = np.sqrt(1 + (eta/(E + self.B0))**2)
        sgn = np.sign(E - self.E0)


        sqrt_w = np.sqrt(self.E_so**2 + 2.0 * self.E_so * z + self.B0**2)
        kp = np.sqrt(2.0 * self.m * (z + self.E_so + sqrt_w))
        km = np.sqrt(2.0 * self.m * (z + self.E_so - sqrt_w))
        
        if(np.imag(kp) > 0):
            k1 =   kp
        else:
            k1 = - kp

        if(np.imag(km) > 0):
            k2 =   km
        else: 
            k2 = - km

        vec = np.array([kp, -kp, km, -km])
        plt.plot(np.real(vec), np.imag(vec), 'b.')

        return k1, k2

    def kp(self, z):
        E = np.real(z)
        eta = np.imag(z)
        inner_sqrt = np.sqrt(1.0 + (eta/(E - self.E0))**2)
        
        w = self.E_so**2 + 2 * self.E_so * z + self.B0**2
        sqrt_w = np.sqrt(w)
        
        return np.sqrt(2.0 * self.m * (E + 1j * eta + self.E_so + sqrt_w))

    def km(self, z):
        E = np.real(z)
        eta = np.imag(z)
        inner_sqrt = np.sqrt(1.0 + (eta/(E - self.E0))**2)
        
        w = self.E_so**2 + 2 * self.E_so * z + self.B0**2
        sqrt_w = np.sqrt(w) 
        
        return np.sqrt(2.0 * self.m * (E + 1j * eta + self.E_so - sqrt_w))

    def find_k2(self, z):
        if(np.imag(self.kp(z)) > 0):
            k1 =   self.kp(z)
        else:
            k1 = - self.kp(z)

        if(np.imag(self.km(z)) > 0):
            k2 =   self.km(z)
        else: 
            k2 = - self.km(z)


        return k1, k2

    def D(self, z, k):
        ksq_2m = (k**2)/(2.0 * self.m)
        return (z - ksq_2m)**2 - 2 * self.E_so * ksq_2m - self.B0**2

    def D_prim(self, z, k):
        return k**3/(self.m**2) - 2.0 * k/self.m * (z + self.E_so)

    def z_cross_R(self, R):
        return np.array([-R[1], R[0]])

    def N(self, E):
        z = E + 1j * self.eta
        R = np.array([self.R_to_0, self.R_to_0])
        return - 1.0/np.pi * np.imag( np.trace(self.G(R, z)))

    def Npl(self, E):
        z = E + 1j * self.eta
        R = np.array([self.R_to_0, self.R_to_0])
        return - 1.0/np.pi * np.imag( np.trace(self.Gp(R, z)))

    def Nmi(self, E):
        z = E + 1j * self.eta
        R = np.array([self.R_to_0, self.R_to_0])
        return - 1.0/np.pi * np.imag( np.trace(self.Gm(R, z)))

    def Gp(self, R, z):
        k1, _= self.find_k2(z)
        R_hat = R / la.norm(R)

        return  0.5 * 1j * np.real(k1) /((self.D_prim(z, k1))) * np.real(sf.hankel1(0, k1 * la.norm(R))) \
                * (z - (k1**2)/(2.0 * self.m)) * self.sigma_0
        
    def Gm(self, R, z):
        _, k2 = self.find_k2(z)
        R_hat = R / la.norm(R)

        return 0.5 * 1j * np.real(k2) /(self.D_prim(z, k2)) * np.real(sf.hankel1(0, k2 * la.norm(R))) \
                 * (z - k2**2/(2.0 * self.m)) * self.sigma_0

    def G(self, R, z):
        k1, k2 = self.find_k2(z)
        R_hat = R / la.norm(R)
        ZxRpB = self.alpha * self.z_cross_R(R_hat) + self.beta * R_hat
        

        res = 0 * 1j
        res += 0.5 * 1j * np.real(k1) /((self.D_prim(z, k1))) * np.real(sf.hankel1(0, k1 * la.norm(R))) \
                * (z - (k1**2)/(2.0 * self.m)) * self.sigma_0
        
        res += 0.5 * 1j * np.real(k2) /(self.D_prim(z, k2)) * np.real(sf.hankel1(0, k2 * la.norm(R))) \
                 * (z - k2**2/(2.0 * self.m)) * self.sigma_0

        # res  = 0.5 * 1j * np.abs(k1)/self.D_prim(z, k1) * sf.hankel1(0, k1 * la.norm(R)) \
                # * (( z - k1*k1/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        # res -= 0.5      * np.abs(k1)/self.D_prim(z, k1) * sf.hankel1(1, k1 * la.norm(R)) \
                # * k1 * (ZxRpB[0] * self.sigma_x + ZxRpB[1] * self.sigma_y)
        # res += 0.5 * 1j * np.abs(k2)/self.D_prim(z, k2) * sf.hankel1(0, k2 * la.norm(R)) \
                # * (( z - k2*k2/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        # res -= 0.5      * np.abs(k2)/self.D_prim(z, k2) * sf.hankel1(1, k2 * la.norm(R)) \
                # * k2 * (ZxRpB[0] * self.sigma_x + ZxRpB[1] * self.sigma_y)
        return res



