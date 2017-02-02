from mpi4py import MPI
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
#from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity

# def plot_den(g,i):
	# global X,Y,Z
	# subplot = 0
	# f, axarr = plt.subplots(3, figsize=(8,12))
	# for E in [-1.0, 0.0, 1.0]:
		# for j in range(X.shape[0]):
			# for k in range(X.shape[1]):
				# r[0] = X[j,k]
				# r[1] = Y[j,k]
				# Z[j,k] = g.dRoh(r, E)
		# lvls = np.linspace(np.min(Z), np.max(Z), 20)
		# cont = axarr[subplot].contourf(X,Y,Z, cmap=cm.coolwarm, levels=lvls)
		# f.colorbar(cont, ax=axarr[subplot])
		# axarr[subplot].set_ylabel("Set: %d  E = %2.2f"%(i+1,E))
		# subplot += 1
	# plt.savefig("plots/task2_circle_den_%03d.pdf"%(i+1))
	# plt.close()
	# print("%3d density done."%(i+1))

def calc_den_distr(g,i,E, xpart, ypart, x_num, y_num, rank):
	subplot = 0
	# if rank == 0:
		# f, axarr = plt.subplots(3, figsize=(8,12))
	# else:
		# f = axarr = None
	for E in [-1.0, 0.0, 1.0]:
		zpart = calc_den(g, xpart, ypart, E)
		X = recomb_grid(x_num, y_num, rank, xpart)
		Y = recomb_grid(x_num, y_num, rank, ypart)
		Z = recomb_grid(x_num, y_num, rank, zpart)
		if rank == 0:
			return X,Y,Z
		else:
			return None, None, None
			#lvls = np.linspace(np.min(Z), np.max(Z), 20)
			# cont = axarr[subplot].contourf(X,Y,Z, cmap=cm.coolwarm, levels=lvls)
			# f.colorbar(cont, ax=axarr[subplot])
			# axarr[subplot].set_ylabel("Set: %d  E = %2.2f"%(i+1,E))
			# subplot += 1
	# if rank == 0:
		# plt.savefig("plots/test_den_%03d.pdf"%(i+1))
		# plt.close()
		# print("%3d density done."%(i+1))


def recomb_grid(x_num, y_num, rank, part):
	grid = comm.gather(part, root=0)
	if rank == 0:
		grid = np.hstack(grid)
		grid = grid.reshape(x_num, y_num)
		return grid
	else:
		return None


# def plot_mag(g,i):
	# global X,Y,Z,U,V,W
	# z_ratio = np.zeros(X.shape)
	# z_vec = np.array([0.0, 0.0, 1.0])
	# subplot = 0
	# Z.fill(0.0)
	# length = np.zeros(Z.shape)
	# f, axarr = plt.subplots(3, figsize=(8,12))
	# for E in [-1.0, 0.0, 1.0]:
		# for j in range(X.shape[0]):
			# for k in range(X.shape[1]):
				# r[0] = X[j,k]
				# r[1] = Y[j,k]
				# tmp = g.dMs(r, E)
				
				# U[j,k] = tmp[0]
				# V[j,k] = tmp[1]
				# W[j,k] = tmp[2]
				# z_ratio[j,k] = W[j,k]/np.sqrt(tmp[0]**2 + tmp[1]**2 + tmp[2]**2)
		# print("min: %f max: %f"%(np.min(z_ratio), np.max(z_ratio)))
		# strm = axarr[subplot].streamplot(X,Y,U,V, color=z_ratio,
				# cmap=cm.viridis)
		# #strm.clim(-1,1)
		# f.colorbar(strm.lines, ax=axarr[subplot])
		# subplot += 1
	# plt.savefig("plots/task2_circle_mag_%03d.pdf"%(i+1))
	# plt.close()
	# print("%3d magnetization done."%(i+1))

def calc_den(g, x_part, y_part, E):
	r      = np.array([0.0, 0.0])
	z_part = np.zeros(x_part.shape)

	for j in range(x_part.shape[0]):
		r[0]      = x_part[j]
		r[1]      = y_part[j]
		z_part[j] = g.dRoh(r, E)
	return z_part

def calc_mag(g, X, Y, E):
	r       = np.array([0.0, 0.0])
	U       = np.zeros(X.shape)
	V       = np.zeros(X.shape)
	W       = np.zeros(X.shape)
	z_ratio = np.zeros(X.shape)

	for j in range(X.shape[0]):
		for k in range(X.shape[1]):
			r[0] = X[j,k]
			r[1] = Y[j,k]
			tmp  = g.dMs(r, E)
			
			U[j,k] = tmp[0]
			V[j,k] = tmp[1]
			W[j,k] = tmp[2]
			z_ratio[j,k] = W[j,k]/np.sqrt(tmp[0]**2 + tmp[1]**2 + tmp[2]**2)
	return U, V, W, z_ratio



def split_grid(x,y, nprocs):
	X, Y = np.meshgrid(x,y)
	xf   = X.flatten()
	yf   = Y.flatten()
	xsp  = np.array_split(xf, nprocs)
	ysp  = np.array_split(yf, nprocs)
	return xsp, ysp 



comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

# init values
N = 8
V = 0.23 * np.ones(N)
#R = np.array([[0.0, 0.5]])
R = np.array([[0.0,    1.0],
			  [-0.781, 0.623],
			  [-0.974, -0.222],
			  [-0.433, -0.900],
			  [0.433,  -0.900],
			  [0.974,  -0.222],
			  [0.781,  0.623],
			  [-0.5, 0.0]])

B      = np.zeros((N,3))
#B[:,0] = V[0]

I      = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(5)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])


dim_num = 150
if rank == 0:
	x = y = np.linspace(-1.2, 1.2, dim_num)
	xsp, ysp = split_grid(x,y, nprocs)
else:
	x = y = xsp = ysp = None

x_part = comm.scatter(xsp, root=0)
y_part = comm.scatter(ysp, root=0)

for i in range(alpha.shape[0]):
	g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I)
	plot_den_distr(g,i, x_part, y_part, dim_num, dim_num, rank)

# x = y = np.linspace(-1.2, 2.2, 150)
# X, Y = np.meshgrid(x,y)
# U = np.zeros(X.shape)
# V = np.zeros(X.shape)
# W = np.zeros(X.shape)
# Z = np.zeros(X.shape)
# r = np.array([0.0, 0.0])
# for i in range(alpha.shape[0]):
        # g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I)
        # plot_mag(g,i)
