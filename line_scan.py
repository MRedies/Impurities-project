import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity
from scipy.optimize import curve_fit

def dE_dk(g,E):
    h = 1e-3
    k1l, k2l = g.find_ks(E - h + g.eta)
    k1r, k2r = g.find_ks(E + h + g.eta)

    dk1 = (k1r - k1l) / (2.0 * h)
    dk2 = (k2r - k2l) / (2.0 * h)
    dE1 = 1.0/dk1
    dE2 = 1.0/dk2

    print("dE/dk1 = %f + i %f"%(np.real(dE1), np.imag(dE1)))
    print("dE/dk2 = %f + i %f"%(np.real(dE2), np.imag(dE2)))

def plot_k(g,Epick, ax, f):
    E  = np.linspace(-10, 10, 10000)
    k1 = np.zeros(E.shape, dtype=np.complex_)
    k2 = np.zeros(E.shape, dtype=np.complex_)
    
    for i in range(E.shape[0]):
        k1[i], k2[i] = g.find_ks(E[i] + g.eta)

    k1p, k2p = g.find_ks(Epick + g.eta)
    print("E = %f"%(Epick))
    print("k1 = %f + i %f"%(np.real(k1p), np.imag(k1p)))
    print("k2 = %f + i %f"%(np.real(k2p), np.imag(k2p)))
    dE_dk(g, Epick)

    ax.plot(E, np.real(k1), label="Re[k1]")
    ax.plot(E, np.real(k2), label="Re[k2]")
    ax.plot(E, np.imag(k1),"-", label="Im[k1]")
    ax.plot(E, np.imag(k2),"-",  label="Im[k2]")
    ax.set_xlabel("E")
    ax.set_ylabel("k")
    ax.legend()

def plot_line(g):
    x   = np.linspace(-1.5, 1.5, 300)
    roh = line(g, x)
    
    plt.plot(x, roh)
    plt.show()

def line(g,x):
    global E
    roh = np.zeros(x.shape)
    r      = np.array([0.0, 0.5])

    for i in range(x.shape[0]):
        r[0]   = x[i]
        roh[i] = g.dRoh(r, E)
    return roh
    
def func(x, a, b, c):
    return a * np.cos(b * x +c) / np.abs(x*x)

def func2(x, a, b,c):
    return a * np.cos(b * x +c)

def fit(g, ax1, ax2):
    x = np.linspace(0.7, 50, 20000)
    y = line(g,x)
    
    g_a = 0.00637
    g_b = 40
    g_c = 0.01
    y2 = func2(x, g_a, g_b, g_c)

    ax1.plot(x, y*x,   '-', label="data * r")
    ax2.plot(x, y*x*x, '-', label="data * r * r")

    #ax.plot(x,y2, label="guess")
    # popt, pcov = curve_fit(func2, x,y, p0=[g_a, g_b, g_c])
    # a = popt[0]
    # b = popt[1]
    # c = popt[2]

    #ax.plot(x, func2(x,a,b,c), label="fit")
    ax1.legend()
    ax2.legend()
    #perr = np.sqrt(np.diag(pcov))


def fft_analysis(g, ax):
    n = 20000
    r = np.linspace(0.7, 20, n)
    y = line(g,r) 
    
    timestep = r[1] - r[0]
    freq = np.fft.fftfreq(n, d =timestep)

    four_r = np.fft.fft(y*r)
    four_r = four_r[freq>=0]
    
    four_rr = np.fft.fft(y*r*r)
    four_rr = four_rr[freq>=0]
    
    freq = freq[freq>=0]
    
    ax.loglog(freq, np.abs(four_r),  label="$FFT[y*r]$")
    ax.loglog(freq, np.abs(four_rr), label="$FFT[y*r^2]$")
    ax.set_xlabel("Freq. $\omega$")
    ax.set_ylabel("Amplitude")
    ax.set_title("Fourier transform")
    ax.legend()

N = 1
V = 0.23 * np.ones(N)
R = np.array([[0.0, 0.5]])
#R = np.array([[0.0,    1.0],
              # [-0.781, 0.623],
              # [-0.974, -0.222],
              # [-0.433, -0.900],
              # [0.433,  -0.900],
              # [0.974,  -0.222],
              # [0.781,  0.623],
              # [-0.5, 0.0]])

B      = np.zeros((N,3))
#B[:,0] = V[0]

I      = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(5)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])

for i in range(3,5):
    for E in [-6.0, -4.0, -2.0]:
        g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I, eta=1e-6, R_to_0=1e-3)

        f, axarr = plt.subplots(2,2)
        axarr = axarr.flatten()
        plot_k(g,E, axarr[0],f)

        a,b = axarr[0].get_ylim()
        axarr[0].plot([E,E],[a,b], "k:",label="E")

        axarr[0].set_title("Set: %d E = %2.2f"%(i+1, E))

        fit(g, axarr[2], axarr[3])
        fft_analysis(g, axarr[1])
        plt.legend()
        plt.show()
