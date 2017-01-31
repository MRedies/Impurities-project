import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity

def plot_den(g,i):
    global X,Y,Z
    subplot = 0
    f, axarr = plt.subplots(3, figsize=(8,12))
    for E in [-1.0, 0.0, 1.0]:
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                r[0] = X[j,k]
                r[1] = Y[j,k]
                Z[j,k] = g.dRoh(r, E)
        cont = axarr[subplot].contourf(X,Y,Z, cmap=cm.viridis)
        f.colorbar(cont, ax=axarr[subplot])
        axarr[subplot].set_ylabel("Set: %d  E = %2.2f"%(i+1,E))
        subplot += 1
    plt.savefig("plots/task2_den_circ_%03d.pdf"%(i+1))
    plt.close()
    print("%3d density done."%(i+1))

def plot_mag(g,i):
    global X,Y,Z,U,V,W
    subplot = 0
    Z.fill(0.0)
    length = np.zeros(Z.shape)
    for E in [-1.0, 0.0, 1.0]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                r[0] = X[j,k]
                r[1] = Y[j,k]
                tmp = g.dMs(r, E)
                
                length[j,k] = la.norm(tmp)
                U[j,k] = tmp[0]
                V[j,k] = tmp[1]
                W[j,k] = tmp[2]
        q = ax.quiver(X,Y,Z,U,V,W, cmap=cm.brg, length=0.2, 
                pivot='middle', clim=[-1,1])
        q.set_array((W/length ).flatten())
        fig.colorbar(q)
        ax.set_zlim([-1,1])
        ax.set_title("Set %d E = %2.2f"%(i, E))
        plt.savefig("plots/task2_mag_circ_%03d_E_%2.2f.pdf"%(i+1, E))
        plt.close()
    print("%3d magnetization done."%(i+1))

N = 7
V = 0.23 * np.ones(N)
R = np.array([[0.0,    1.0],
              [-0.781, 0.623],
              [-0.974, -0.222],
              [-0.433, -0.900],
              [0.433,  -0.900],
              [0.974,  -0.222],
              [0.781,  0.623]])
B = np.zeros((N,3))
I = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(N)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])

x = y = np.linspace(-1.2, 1.2, 150)
X, Y = np.meshgrid(x,y)
U = np.zeros(X.shape)
V = np.zeros(X.shape)
W = np.zeros(X.shape)
Z = np.zeros(X.shape)
r = np.array([0.0, 0.0])
for i in range(alpha.shape[0]):
    g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I)
    #plot_den(g,i)


x = y = np.linspace(-1.2, 1.2, 10)
X, Y = np.meshgrid(x,y)
U = np.zeros(X.shape)
V = np.zeros(X.shape)
W = np.zeros(X.shape)
Z = np.zeros(X.shape)
r = np.array([0.0, 0.0])
for i in range(alpha.shape[0]):
    g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I)
    plot_mag(g,i)
