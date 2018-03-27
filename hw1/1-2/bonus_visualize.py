import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.cm as cm

params = np.load('bonus_params.npy')
loss =  np.load('bonus_loss.npy')
loss = (loss - np.mean(loss))/np.std(loss)
color =  np.load('bonus_colors.npy')
loss_p = np.array([ np.percentile(loss, pr) for pr in np.arange(10,100,10)])
dim = 2
'''
print('start PCA')
out = PCA(10).fit_transform(params)
print('start TSNE')
out = TSNE(dim).fit_transform(out)
print('finished TSNE')
#tsne = TSNE(dim)
'''
out = np.load('pca_tsne_out' + str(dim) +'.npy')
#np.save('tsne_out' + str(dim),out)
#np.save('pca_tsne_out' + str(dim),out)

print('finish fitting')
#print(out.shape)
#exit()
dim = 3
if dim == 3: # 3D
	fig = plt.figure()
	ax = Axes3D(fig)


	XX = []
	YY = []
	ZZ = []
	for idx in range(color.shape[0]):
		if color[idx] == 1:
			XX.append(out[idx][0])
			YY.append(out[idx][1])
			#ZZ.append(out[idx][2])
			ZZ.append(loss[idx])
	ax.plot(XX,YY,ZZ,color='green')
	X = out[:,0]
	Y = out[:,1]
	#Z = out[:,2]
	Z = loss

	finx , finy , finz = [],[],[]
	for x,y,z,l in zip(X,Y,Z,loss):
		# drop 90%
		if np.random.uniform(0,1) > 0.01:
			red = len(loss_p[loss_p < l])
			#ax.scatter(x, y, z , color=[(red/10 , 0 , 1-red/10)])
			finx.append(x)
			finy.append(y)
			finz.append(z)

	#ax.scatter(X,Y,Z)
	#ax.set_zlim(0,500)
	finx = np.array(finx)
	finy = np.array(finy)
	finz = np.array(finz)
	from scipy.interpolate import griddata
	grid_x, grid_y = np.mgrid[min(finx):max(finx):100j, min(finy):max(finy):200j]
	points = np.array([ [x,y] for x,y in zip(finx,finy) ])
	print(finz.shape , points.shape)
	grid_z = griddata(points, finz, (grid_x,grid_y) ,method='nearest')
	print(grid_z)

	for i in grid_z:
		for j in i :
			print(j , end=',')
	ax.plot_surface(grid_x, grid_y, grid_z, color = 'gray', rstride=1, cstride=1, linewidth=0)
	plt.savefig('fig')
	plt.show()

if dim == 2: #2D
	XX = []
	YY = []
	for idx in range(color.shape[0]):
		if color[idx] == 1:
			XX.append(out[idx][0])
			YY.append(out[idx][1])
	
	plt.plot(XX,YY,color='black')
	X = out[:,0]
	Y = out[:,1]
	for x,y,l in zip(X,Y,loss):
		red = len(loss_p[loss_p < l])
		plt.scatter(x, y , color=[(red/10 , 0 , 1-red/10)])
	#ax.scatter(X,Y,Z)
	#ax.set_zlim(0,500)
	plt.show()
