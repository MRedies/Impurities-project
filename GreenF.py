import numpy as np
import scipy.special as sf


def hankel1_small_angle(alpha, z):
    return np.sqrt(2.0/(np.pi * z)) * np.exp(1j * (z - 0.5 * alpha * np.pi - 0.25 * np.pi))

class GF:
    def __init__(self, m, alpha, beta, B0, eta = 1e-5):
        self.m     = complex(m,     0)
        self.alpha = complex(alpha, 0)
        self.beta  = complex(beta,  0)
        self.B0    = complex(B0,    0)
        self.eta   = eta
        self.E_so  = self.m * (alpha**2 + beta**2)
        self.E0    = - (self.E_so**2 + self.B0**2)/(2 * self.E_so)

        self.sigma_0 = np.identity(2)
        self.sigma_x = np.array([[0, 1],[1, 0]])
        self.sigma_y = np.array([[0, -1j],[1j, 0]])
        self.sigma_z = np.array([[1, 0],[0, -1]])

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
        R = np.array([1e-6, 1e-6])
        return - 1.0/np.pi * np.imag( np.trace(self.G(R, z)))

    def N_small_angle(self, E):
        z = E + 1j * self.eta
        R = np.array([1e-6, 1e-6])
        return - 1.0/np.pi * np.imag( np.trace(self.G_small_angle(R, z)))

    def G_small_angle(self, R, z):
        k1, k2 = self.find_k(z)
        ZxR = self.alpha * self.z_cross_R(R) + self.beta * R
        res  = 0.5 * 1j * np.abs(k1)/self.D_prim(z, k1) * hankel1_small_angle(0, k1 * np.abs(R)) \
                * (( z - k1*k1/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        res -= 0.5      * np.abs(k1)/self.D_prim(z, k1) * hankel1_small_angle(1, k1 * np.abs(R)) \
                * k1 * (ZxR[0] * self.sigma_x + ZxR[1] * self.sigma_y)
        res += 0.5 * 1j * np.abs(k2)/self.D_prim(z, k2) * hankel1_small_angle(0, k2 * np.abs(R)) \
                * (( z - k2*k2/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        res -= 0.5      * np.abs(k2)/self.D_prim(z, k2) * hankel1_small_angle(1, k2 * np.abs(R)) \
                * k2 * (ZxR[0] * self.sigma_x + ZxR[1] * self.sigma_y)
        return res

    def G(self, R, z):
        k1, k2 = self.find_k(z)
        ZxR = self.alpha * self.z_cross_R(R) + self.beta * R
        res  = 0.5 * 1j * np.abs(k1)/self.D_prim(z, k1) * sf.hankel1(0, k1 * np.abs(R)) \
                * (( z - k1*k1/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        res -= 0.5      * np.abs(k1)/self.D_prim(z, k1) * sf.hankel1(1, k1 * np.abs(R)) \
                * k1 * (ZxR[0] * self.sigma_x + ZxR[1] * self.sigma_y)
        res += 0.5 * 1j * np.abs(k2)/self.D_prim(z, k2) * sf.hankel1(0, k2 * np.abs(R)) \
                * (( z - k2*k2/(2*self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        res -= 0.5      * np.abs(k2)/self.D_prim(z, k2) * sf.hankel1(1, k2 * np.abs(R)) \
                * k2 * (ZxR[0] * self.sigma_x + ZxR[1] * self.sigma_y)
        return res

