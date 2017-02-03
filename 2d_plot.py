#/usr/bin/ipython
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity


def calc_den_distr(g,E, xpart, ypart, x_num, y_num, rank):
        subplot = 0
        for E in [-1.0, 0.0, 1.0]:
                zpart = calc_den_loc(g, xpart, ypart, E)
                X = recomb_grid(x_num, y_num, rank, xpart)
                Y = recomb_grid(x_num, y_num, rank, ypart)
                Z = recomb_grid(x_num, y_num, rank, zpart)
                if rank == 0:
                        return X,Y,Z
                else:
                        return None, None, None

def calc_den_loc(g, x_part, y_part, E):
    r      = np.array([0.0, 0.0])
    z_part = np.zeros(x_part.shape)

    for j in range(x_part.shape[0]):
        r[0]      = x_part[j]
        r[1]      = y_part[j]
        z_part[j] = g.dRoh(r, E)
    return z_part

def plot_den(X,Y,Z, Set, E, ax, f):
    lvls = np.linspace(np.min(Zarr[i]), np.max(Zarr[i]), 20)
    cont = ax.contourf(X, Y, Z, cmap=cm.viridis, levels=lvls)
    f.colorbar(cont, ax=ax)
    ax.set_ylabel("Set: %d, E = %2.2f"%(Set+1, E))

def plot_k(g,ax, f):
    E  = np.linspace(-10, 10, 100)
    k1 = np.zeros(E.shape, dtype=np.complex_)
    k2 = np.zeros(E.shape, dtype=np.complex_)
    
    for i in range(E.shape[0]):
        k1[i], k2[i] = g.find_ks(E[i] + g.eta)
    ax.plot(E, np.real(k1), label="k1")
    ax.plot(E, np.real(k2), label="k2")
    ax.set_xlabel("E")
    ax.set_ylabel("k")
    ax.legend()

def create_den_plot(g,x,y, Set, E, comm):
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]

    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(2,2, figsize=(8,8))
        axarr_flat = [axarr[1,0], axarr[0,1], axarr[1,1]]
        plot_k(g, axarr[0,0], f)
        a,b = axarr[0,0].get_ylim()
        axarr[0,0].plot([E,E],[a,b], "k:",label="E")
        axarr[0,0].legend()
    
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)

    
    for i in range(3):
        X, Y, Z = calc_den_distr(g,E[i], x_part, y_part,
                dim_num, dim_num, rank)
        if rank == 0:
            lvls = np.linspace(np.min(Z), np.max(Z), 20)
            cont = axarr_flat[i].contourf(X,Y,Z, cmap=cm.coolwarm, levels=lvls)
            f.colorbar(cont, ax=axarr_flat[i])
            axarr_flat[i].set_title("Set: %d  E = %2.2f"%(Set+1,E[i]))
        
    if rank == 0:
        plt.show()

def SplitSpread_grid(x,y, comm):
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        X, Y = np.meshgrid(x,y)
        xf   = X.flatten()
        yf   = Y.flatten()
        xsp  = np.array_split(xf, nprocs)
        ysp  = np.array_split(yf, nprocs)
    else:
        x = y = xsp = ysp = None
    x_part = comm.scatter(xsp, root=0)
    y_part = comm.scatter(ysp, root=0)
    return x_part, y_part


def recomb_grid(x_num, y_num, rank, part):
	grid = comm.gather(part, root=0)
	if rank == 0:
		grid = np.hstack(grid)
		grid = grid.reshape(x_num, y_num)
		return grid
	else:
		return None




def calc_mag_loc(g, x_part, y_part, E):
    r       = np.array([0.0, 0.0])
    u       = np.zeros(x_part.shape)
    v       = np.zeros(x_part.shape)
    w       = np.zeros(x_part.shape)
    z_ratio = np.zeros(x_part.shape)

    for j in range(x_part.shape[0]):
        r[0] = x_part[j]
        r[1] = y_part[j]
        tmp  = g.dMs(r, E)
        
        u[j] = tmp[0]
        v[j] = tmp[1]
        w[j] = tmp[2]
        z_ratio[j] = w[j]/np.sqrt(tmp[0]**2 + tmp[1]**2 + tmp[2]**2)
    return u, v, w, z_ratio


def calc_mag_distr(g,E, xpart, ypart, x_num, y_num, rank):
        subplot = 0
        for E in [-1.0, 0.0, 1.0]:
                u,v,w,z_ratio = calc_mag_loc(g, xpart, ypart, E)
                X = recomb_grid(x_num, y_num, rank, xpart)
                Y = recomb_grid(x_num, y_num, rank, ypart)
                U = recomb_grid(x_num, y_num, rank, u)
                V = recomb_grid(x_num, y_num, rank, v)
                W = recomb_grid(x_num, y_num, rank, w)

                Z_ratio = recomb_grid(x_num, y_num, rank, z_ratio)
                if rank == 0:
                        return X,Y,U,V,W,Z_ratio
                else:
                        return None, None, None, None, None, None

def create_mag_plot_stream(g,x,y, Set, E, comm):
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]

    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(2,2, figsize=(8,8))
        axarr_flat = [axarr[1,0], axarr[0,1], axarr[1,1]]
        plot_k(g, axarr[0,0], f)
        a,b = axarr[0,0].get_ylim()
        axarr[0,0].plot([E,E],[a,b], "k:",label="E")
        axarr[0,0].legend()
    
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)

    
    for i in range(3):
        X, Y, U, V, W, Z_ratio = calc_mag_distr(g,E[i], x_part, y_part,
                dim_num, dim_num, rank)
        if rank == 0:
            strm = axarr_flat[i].streamplot(X,Y,U,V, color=Z_ratio,
                            cmap=cm.viridis)
            f.colorbar(strm.lines, ax=axarr_flat[i])
            axarr_flat[i].set_title("Set: %d  E = %2.2f"%(Set+1,E[i]))
            axarr_flat[i].set_xlim(np.min(X), np.max(X))
            axarr_flat[i].set_ylim(np.min(Y), np.max(Y))
    
    if rank == 0:
        plt.show()

def create_mag_plot_contour(g,x,y, Set, E, comm):
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]


    for i in range(3):
        #create k-plot
        if rank == 0:
            f, axarr = plt.subplots(2,2, figsize=(8,8))
            plot_k(g, axarr[0,0], f)
            a,b = axarr[0,0].get_ylim()
            axarr[0,0].plot([E[i],E[i]],[a,b], "k:",label="E")
            axarr[0,0].legend()
        #f.title("Set: %d  E = %2.2f"%(Set+1,E[i]))
        #distrib grid
        x_part, y_part = SplitSpread_grid(x,y, comm)
        X, Y, U, V, W, Z_ratio = calc_mag_distr(g,E[i], x_part, y_part,
                dim_num, dim_num, rank)
        if rank == 0:
            lvls = np.linspace(np.min(U), np.max(U), 20)
            cont = axarr[0,1].contourf(X,Y,U, cmap=cm.coolwarm, levels=lvls)
            f.colorbar(cont, ax=axarr[0,1])
    
            lvls = np.linspace(np.min(V), np.max(V), 20)
            cont = axarr[1,0].contourf(X,Y,V, cmap=cm.coolwarm, levels=lvls)
            f.colorbar(cont, ax=axarr[1,0])
            
            lvls = np.linspace(np.min(W), np.max(W), 20)
            cont = axarr[1,1].contourf(X,Y,W, cmap=cm.coolwarm, levels=lvls)
            f.colorbar(cont, ax=axarr[1,1])
            
            axarr[0,0].set_title("Set: %d  E = %2.2f"%(Set+1,E[i]))
            axarr[0,1].set_title("X-component")
            axarr[1,0].set_title("Y-component")
            axarr[1,1].set_title("Z-component")
            plt.show()

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

# init values
N = 1
V = 0.23 * np.ones(N)
R = np.array([[0.0, 0.5]])
# R = np.array([[0.0,    1.0],
              # [-0.781, 0.623],
              # [-0.974, -0.222],
              # [-0.433, -0.900],
              # [0.433,  -0.900],
              # [0.974,  -0.222],
              # [0.781,  0.623],
              # [-0.5,   0.0]])

B      = np.zeros((N,3))
#B[:,0] = V[0]

I      = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(5)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3 ])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])


dim_num = 150
x = y = np.linspace(-1.2, 1.2, dim_num)


E = np.array([-1.0, 0.0, 1.0])
for i in range(alpha.shape[0]):
        g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I)
        #create_mag_plot_stream(g,x,y,i,E,comm)
        create_den_plot(g,x,y, i, E, comm)
        print("[%2d]: %d"%(rank,i))
