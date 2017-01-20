import numpy as np

class GF:
    def __init__(self, m, alpha, beta, B0):
        self.m     = complex(m,     0)
        self.alpha = complex(alpha, 0)
        self.beta  = complex(beta,  0)
        self.B0    = complex(B0,    0)
        self.E_so  = (alpha**2 + beta**2)
        self.E0    = - (self.E_so**2 + self.B0**2)/(2 * self.E_so)


    def find_k(self, z):
        E   = np.real(z)
        eta = np.imag(z)
        inner_sqrt   = np.sqrt(1 + (eta/(E + self.B0))**2)
        sgn = np.sign(E - self.E0)
        # sqrt_w = np.sqrt(2 * self.E_so * np.abs(E - self.E0)) * \
                 # ( np.sqrt(0.5 * ( inner_sqrt + sgn))
            # + 1j * np.sqrt(0.5 * ( inner_sqrt - sgn)))


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
