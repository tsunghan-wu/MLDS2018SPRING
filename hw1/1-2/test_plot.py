import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
print(np.random.rand(1000, 2).shape)
X, Y = np.mgrid[-1:1:20j, -1:1:20j]
Z = (X+Y) * np.exp(-6.0*(X*X+Y*Y)) + np.random.rand(X.shape[0])
print(X.shape , Y.shape , Z.shape)	
xnew, ynew = np.mgrid[-1:1:80j, -1:1:80j]
tck = interpolate.bisplrep(X, Y, Z, s=0)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='summer', rstride=1, cstride=1, alpha=None)
plt.show()

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')
ax.plot_surface(xnew, ynew, znew, cmap='summer', rstride=1, cstride=1, alpha=None, antialiased=True)
#plt.show()
